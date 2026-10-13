"""Train ARDrift (``train-ar``, #42): an autoregressive acoustic model with a one-step drifting head.

One step:

1. the text encoder and monotonic alignment, prior and duration losses (as ``train``);
2. the frame-aligned condition and the target mel, padded to whole patches; every utterance is used whole;
3. the causal backbone, teacher-forced on the previous target patches plus noise of a random level per utterance
   (up to ``drift.input_noise``), gives one state per patch;
4. at ``drift.max_rows`` random valid patches (rows), the head draws ``drift.gen_per_cond`` samples from noise; the
   drifting loss (:func:`~drifting_tts.drift.kyutai_drift_loss`, one learned temperature) pulls them to the recorded
   patch and pushes them apart, in the kernel space of :func:`~drifting_tts.models.ar.ar_features` (the whole patch,
   each frame, each frame's harmonic detail).
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import torch

from .alignment import sequence_mask
from .config import Config, load_config, save_config
from .data import MelDataset
from .drift import feature_drift_loss
from .models.ar import ARDriftTTS, ar_features, patchify
from .models.text_encoder import align, duration_loss, expand, prior_loss
from .train import build_loader, dataset_stats, grow_speakers
from .utils import EMA, count_params, infinite, lr_lambda, rng_state, save_checkpoint, seed_everything, set_rng_state


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default="configs/ar_tts.yaml")
    p.add_argument("--workdir", default="runs/ar_tts")
    p.add_argument("overrides", nargs="*", help="config overrides, e.g. train.steps=1000")


def training_step(model: ARDriftTTS, batch: dict, cfg: Config, device, tau: torch.Tensor) -> tuple:
    dc, lc = cfg.drift, cfg.loss
    text, text_len, y, y_len, spk = (batch[k].to(device, non_blocking=True)
                                     for k in ("text", "text_len", "mel", "mel_len", "spk"))
    amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=y.is_cuda)
    y_mask = sequence_mask(y_len, y.shape[-1])[:, None].float()
    with amp:
        h, mu, logw, x_mask = model.encoder(text, text_len, spk)
    h, mu, logw = h.float(), mu.float(), logw.float()
    attn, logw_target = align(mu, x_mask, y, y_mask)
    l_prior = prior_loss(expand(mu, attn), y, y_mask)
    l_dur = duration_loss(logw, logw_target, x_mask)

    k, C = model.patch, model.n_mels
    cond, y = model.pad_frames(model.frame_condition(h, mu, attn)), model.pad_frames(y)
    B = y.shape[0]
    noise = torch.rand(B, device=device) * float(dc.get("input_noise", 0.0))
    p_drop = float(dc.get("cond_dropout", 0.0))  # the null condition of guidance
    drop = torch.rand(B, device=device) < p_drop if p_drop > 0 else None
    with amp:
        s = model.teacher_forced(cond, y, spk, noise, drop)  # [B, S, dim]
    S = s.shape[1]
    valid = (torch.arange(S, device=device)[None] * k < y_len[:, None])  # steps that start inside the utterance
    idx = valid.flatten().nonzero()[:, 0]
    if idx.numel() > dc.max_rows:  # a random subset of the steps (rows) keeps the head's batch bounded
        idx = idx[torch.randperm(idx.numel(), device=device)[: dc.max_rows]]
    rows = s.flatten(0, 1)[idx]  # [R, dim]
    target = patchify(y, k).flatten(0, 1)[idx]  # [R, k * C]
    base = patchify(cond[:, :C], k).flatten(0, 1)[idx]  # the aligned prior mean: the head predicts a residual
    R, G = rows.shape[0], int(dc.gen_per_cond)
    z = torch.randn(R * G, model.head.noise_dim, device=device)
    with amp:
        res = model.head(rows.repeat_interleave(G, 0), z)
    x = base[:, None] + res.float().view(R, G, -1)  # [R, G, k * C]
    detail = bool(dc.get("detail", True))
    f_gen = ar_features(x, k, C, detail)
    with torch.no_grad():
        f_pos = ar_features(target[:, None], k, C, detail)
    l_drift, info = feature_drift_loss(f_gen, f_pos, None, taus=tau, reduce="mean",
                                       key_weights=dc.get("key_weights"))
    loss = lc.drift * l_drift + lc.prior * l_prior + lc.duration * l_dur
    with torch.no_grad():
        metrics = {"loss": loss.detach(), "drift": l_drift.detach() - info["tau_loss"], "tau_loss": info["tau_loss"],
                   "prior": l_prior.detach(), "duration": l_dur.detach(),
                   "centroid_mse": ((x.mean(1) - target) ** 2).mean(), "sample_std": x.std(1).mean(),
                   "rows": float(R), "tau": tau.detach().reshape(()),
                   "p_data": torch.stack([v for kk, v in info.items() if kk.startswith("p_data/")]).mean()}
    return loss, metrics


@torch.no_grad()
def log_samples(model: ARDriftTTS, ds: MelDataset, vocoder, out_dir: Path, step: int, device, n: int = 3,
                temperature: float = 0.5) -> None:
    import soundfile as sf

    was = model.training
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(min(n, len(ds))):
        it = ds[i]
        text = it["text"][None].to(device)
        g = torch.Generator(device=device).manual_seed(i)
        mel, _ = model.synthesize(text, torch.tensor([text.shape[1]], device=device),
                                  torch.tensor([it["spk"]], device=device), temperature=temperature, generator=g)
        wav = vocoder(ds.stats.denormalize(mel))[0].cpu()
        sf.write(out_dir / f"step{step}_utt{i}.wav", wav.numpy(), 24000)
    model.train(was)


def run(args) -> None:
    from torch.utils.tensorboard import SummaryWriter

    cfg = load_config(args.config, args.overrides)
    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    save_config(cfg, work / "config.yaml")
    tc = cfg.train
    seed_everything(tc.seed)
    device = "cuda" if torch.cuda.is_available() and not tc.get("cpu", False) else "cpu"

    if cfg.data.get("stream"):  # a corpus larger than the disk, read from its parquet shards (drifting_tts/stream.py)
        from .stream import stream_loader

        ds, loader = stream_loader(cfg)
        val_ds = ds
    else:
        ds, loader = build_loader(cfg)
        val_ds = MelDataset(cfg.data.root, "val", min_frames=1, max_frames=10**9)
    if ds.language != cfg.model.text.get("language", "tr"):
        raise ValueError(f"{cfg.data.root} is {ds.language!r} text (stats.json), the model's text frontend is "
                         f"{cfg.model.text.get('language', 'tr')!r}: set model.text.language")
    model = ARDriftTTS(cfg.model, num_speakers=ds.num_speakers, n_mels=ds.dim).to(device)
    print(f"ARDrift: text encoder {count_params(model.encoder):.2f}M, backbone "
          f"{count_params(model) - count_params(model.encoder) - count_params(model.head):.2f}M, head "
          f"{count_params(model.head):.2f}M; "
          + ("streamed training data" if cfg.data.get("stream") else f"{len(ds)} training utterances"), flush=True)
    tau = torch.nn.Parameter(torch.tensor(float(cfg.drift.get("tau_init", 1.0)), device=device))
    betas = tuple(tc.get("betas", (0.9, 0.999)))
    opt = torch.optim.AdamW([{"params": list(model.parameters())},
                             {"params": [tau], "weight_decay": 0.0}],
                            lr=tc.lr, betas=betas, weight_decay=tc.weight_decay, fused=device == "cuda")
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(tc.warmup, tc.steps, tc.schedule))
    ema = EMA(model, tc.ema_decay)
    writer = SummaryWriter(work / "tb")
    step, rng = 0, None
    if (work / "last.pt").exists():
        ck = torch.load(work / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        ema.model.load_state_dict(ck["ema"])
        tau.data.fill_(float(ck["tau"]))
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        step, rng = ck["step"], ck.get("rng")
        if cfg.data.get("stream"):  # another shard order than before the restart (the stream has no position)
            loader.dataset.epoch = ck.get("step", 0)
        else:
            loader.batch_sampler.epoch = ck.get("epoch", 0)
        print(f"resumed from step {step}", flush=True)
    elif tc.get("init_from"):  # fine-tuning (e.g. a new voice): speaker tables grown, temperature carried over
        init = torch.load(tc.init_from, map_location="cpu", weights_only=False)
        state = grow_speakers(init.get("ema", init.get("model")), model.state_dict())
        missing, unexpected = model.load_state_dict(state, strict=False)
        ema.model.load_state_dict(model.state_dict())
        if init.get("tau") is not None:
            tau.data.fill_(float(init["tau"]))
        print(f"initialised from {tc.init_from} (missing {len(missing)}, unexpected {len(unexpected)}, "
              f"tau {float(tau):.4g})", flush=True)

    vocoder = None
    if tc.get("sample_every", 0) and device == "cuda":
        from .vocoder import load_vocoder

        vocoder = load_vocoder(tc.get("vocoder", "vocos-v2"), device)

    model.train()
    if rng is not None:
        set_rng_state(rng)
    it = infinite(loader)
    t0, agg = time.time(), {}
    while step < tc.steps:
        loss, metrics = training_step(model, next(it), cfg, device, tau)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm_enc = torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), tc.grad_clip)
        gnorm = torch.nn.utils.clip_grad_norm_([p for n, p in model.named_parameters()
                                                if not n.startswith("encoder.")], tc.grad_clip)
        opt.step()
        sched.step()
        with torch.no_grad():
            tau.clamp_(min=1e-3)
        ema.update(model)
        step += 1
        metrics["grad_norm"], metrics["grad_norm_enc"] = gnorm, gnorm_enc
        for kk, v in metrics.items():
            agg[kk] = agg.get(kk, 0.0) + (v.detach() if torch.is_tensor(v) else v)
        if step % tc.log_every == 0:
            n = tc.log_every
            agg = {kk: float(v) / n for kk, v in agg.items()}
            for kk, v in agg.items():
                writer.add_scalar(f"train/{kk}", v, step)
            writer.add_scalar("train/lr", sched.get_last_lr()[0], step)
            print(f"step {step} " + " ".join(f"{kk}={v:.4g}" for kk, v in agg.items())
                  + f" ({n / (time.time() - t0):.2f} it/s)", flush=True)
            agg, t0 = {}, time.time()
        if vocoder is not None and step % tc.sample_every == 0:
            log_samples(ema.model, val_ds, vocoder, work / "samples", step, device,
                        temperature=tc.get("sample_temperature", 0.5))
        if step % tc.save_every == 0 or step == tc.steps:
            save_checkpoint(work / "last.pt", model=model.state_dict(), ema=ema.model.state_dict(),
                            tau=float(tau.detach()), opt=opt.state_dict(), sched=sched.state_dict(), step=step,
                            rng=rng_state(), epoch=getattr(loader.batch_sampler, "epoch", 0), config=cfg.to_dict(),
                            num_speakers=ds.num_speakers)
            save_checkpoint(work / "model_ema.pt", ema=ema.model.state_dict(), config=cfg.to_dict(),
                            num_speakers=ds.num_speakers, step=step, tau=float(tau.detach()), n_mels=ds.dim,
                            stats=dataset_stats(ds))
            if tc.get("keep_snapshots", False):
                shutil.copyfile(work / "model_ema.pt", work / f"model_ema_{step}.pt")
    print(f"done: {work / 'model_ema.pt'}", flush=True)
    (work / "done.json").write_text(json.dumps({"step": step}))
