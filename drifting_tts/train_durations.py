"""Re-fit a trained model's duration predictor to the mean duration of every token (``train-durations``).

``train`` and ``train-ar`` regress log-durations (:func:`~drifting_tts.models.text_encoder.duration_loss`), so the
predictor learns ``exp(E[log d])``, the geometric mean. For letters that is close to the mean, because their
durations vary little. For the tokens at punctuation it is not: the same comma is read with no pause by one speaker
and a long one by the next, and the geometric mean of such a spread is a fraction of its mean. The English ARDrift
model (#72) pauses 0.45 s per held-out utterance at punctuation against the recordings' 1.19 s, while its letters are
within 4%. ``calibrate-durations`` scales every token alike, so it cannot lengthen the pauses without slowing the
letters.

This command trains only the duration predictor, with everything else frozen and the text encoder in eval mode, on
the model's own MAS durations. It uses the Poisson loss ``exp(logw) - t * logw``, which is minimised at
``exp(logw) = E[t]``, with ``t = d - 0.5``: synthesis rounds every duration up
(:func:`~drifting_tts.models.text_encoder.durations_to_alignment`), which adds half a frame on average. Without the
offset, the ~150 tokens of an English sentence (letters and the blanks between them) come out 0.8 s too long.

The sentence's edges (its first 3 and last 2 tokens) keep the log loss (``--edges log``). Their means carry how the
clips were cut. The Turkish female voice's clips start right at the first word, so the re-fitted lead-in fell to 1
frame and its first words were garbled: Freya-495 WER 3.53% -> 4.32%, mostly a syllable put before the first word.
Its clips also end with long silences, which doubled the final pause.

The output is a copy of the checkpoint with the new predictor; ``durations`` records how it was fitted. Per-voice
duration factors from ``calibrate-durations`` are dropped, since they were measured on the old predictor: run that
command again on the output if wanted.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .alignment import sequence_mask
from .data import BucketBatchSampler, MelDataset, collate
from .models.text_encoder import align
from .utils import infinite, lr_lambda, save_checkpoint, seed_everything

LOSSES = ("poisson", "log")


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True, help="exported model (model_ema.pt or a released .pt)")
    p.add_argument("--out", required=True, help="output checkpoint (the model with the re-fitted predictor)")
    p.add_argument("--data", default=None, help="data root (default: the model's data.root)")
    p.add_argument("--loss", choices=LOSSES, default="poisson",
                   help="poisson: the mean duration (minus the half frame that rounding up adds); log: the training "
                        "loss (the geometric mean), for comparison")
    p.add_argument("--edges", choices=("log", "mean"), default="log",
                   help="the sentence's first 3 and last 2 tokens: log keeps the training loss (geometric mean) there; "
                        "their means carry the clips' cutting (a start right at the first word, long trailing "
                        "silences), which garbles first words")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-frames", type=int, default=None, help="mel frames per batch (default: the model's)")
    p.add_argument("--batch-size", type=int, default=None, help="utterances per batch (default: the model's)")
    p.add_argument("--val", type=int, default=200, help="val utterances measured before and after (0: none)")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")


def duration_targets(model, batch: dict, device) -> tuple:
    """Frozen text encoder and MAS: the predictor's input ``[B, d + spk_dim, N]``, the token mask ``[B, 1, N]`` and
    the MAS durations in frames ``[B, N]``."""
    text, text_len, y, y_len, spk = (batch[k].to(device, non_blocking=True)
                                     for k in ("text", "text_len", "mel", "mel_len", "spk"))
    with torch.no_grad():
        h, mu, _, x_mask = model.encoder(text, text_len, spk)
        attn, _ = align(mu, x_mask, y, sequence_mask(y_len, y.shape[-1])[:, None].float())
        s = model.encoder.spk(spk)
        dur_in = torch.cat([h, s[:, :, None].expand(-1, -1, h.shape[-1])], 1)
    return dur_in, x_mask, attn.sum(-1)


def duration_fit_loss(logw: torch.Tensor, frames: torch.Tensor, x_mask: torch.Tensor, kind: str,
                      edges: bool = False) -> torch.Tensor:
    """``logw`` ``[B, 1, N]`` against the MAS frames ``[B, N]`` (>= 1 on real tokens), mean over tokens. ``edges``: the
    sentence's first 3 and last 2 tokens (:func:`~drifting_tts.models.prosody_net.edge_tokens`) keep the log loss of
    training, whatever ``kind`` is."""
    m = x_mask[:, 0]
    logw = logw[:, 0]
    log_loss = (logw - torch.log(frames.clamp_min(1e-8))) ** 2
    if kind == "poisson":  # the mean of d - 0.5: synthesis rounds up
        per = torch.exp(logw) - (frames - 0.5).clamp_min(0) * logw
    elif kind == "log":
        per = log_loss
    else:
        raise ValueError(f"loss must be one of {LOSSES}, got {kind!r}")
    if edges:
        from .models.prosody_net import edge_tokens

        per = torch.where(edge_tokens(m.sum(1).long(), m.shape[1]), log_loss, per)
    return (per * m).sum() / m.sum()


@torch.no_grad()
def length_ratios(model, ds: MelDataset, num: int, device) -> np.ndarray:
    """Recorded / predicted length (frames) of the first ``num`` utterances of ``ds``."""
    from .models.text_encoder import durations_to_alignment

    out = []
    for i in range(min(num, len(ds))):
        it = ds[i]
        text = it["text"][None].to(device)
        _, _, logw, x_mask = model.encoder(text, torch.tensor([text.shape[1]], device=device),
                                           torch.tensor([it["spk"]], device=device))
        out.append(it["mel"].shape[1] / int(durations_to_alignment(logw, x_mask)[1]))
    return np.array(out)


def duration_choices(model, path: str | Path, new: str, old: str) -> dict[str, dict]:
    """Two sets of durations for one loaded model, to switch per request (a Space, :func:`use_durations`): ``new``,
    the predictor of ``path`` (``publish.export_durations``) with its own per-voice factors (none after
    ``train-durations``), and ``old``, the model's own predictor and factors."""
    import copy

    ck = torch.load(path, map_location="cpu", weights_only=True)
    fitted = copy.deepcopy(model.encoder.duration)
    fitted.load_state_dict(ck["duration"])
    return {new: {"module": fitted.eval(), "scale": float(ck.get("duration_scale", 1.0)),
                  "scales": {int(k): float(v) for k, v in ck.get("duration_scales", {}).items()}},
            old: {"module": model.encoder.duration, "scale": float(model.duration_scale),
                  "scales": dict(model.duration_scales)}}


def use_durations(model, choice: dict) -> None:
    """Point ``model`` at one entry of :func:`duration_choices`: its duration predictor and per-voice factors."""
    model.encoder.duration = choice["module"]
    model.duration_scale, model.duration_scales = choice["scale"], choice["scales"]


def run(args) -> None:
    from .train import load_tts

    seed_everything(args.seed)
    model, cfg, _ = load_tts(args.model, args.device)
    d, tc = cfg.data, cfg.train
    root = args.data or d.root
    ds = MelDataset(root, "train", min_quality=d.get("min_quality", 0.0), max_frames=d.get("max_frames", 1500),
                    filters=d.get("filters"))
    val = MelDataset(root, "val", min_frames=1, max_frames=10**9) if args.val else None
    n_spk = model.encoder.spk.num_embeddings  # a corpus that grew after the model (merge-data): its speakers only
    for split in (ds, val):
        if split is not None and any(e["spk_id"] >= n_spk for e in split.items):
            kept = [e for e in split.items if e["spk_id"] < n_spk]
            print(f"{len(split.items) - len(kept)} utterances of speakers beyond the model's {n_spk} left out",
                  flush=True)
            split.items = kept
    if val is not None and not len(val):
        val = None
    sampler = BucketBatchSampler([ds.frames(i) for i in range(len(ds))],
                                 max_frames=args.batch_frames or tc.get("batch_frames", 24000),
                                 max_batch=args.batch_size or tc.get("batch_size", 24), seed=args.seed)
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate, num_workers=args.num_workers,
                        pin_memory=True, persistent_workers=args.num_workers > 0)
    model.eval().requires_grad_(False)
    dp = model.encoder.duration
    dp.train().requires_grad_(True)
    if val is not None:
        r = length_ratios(model, val, args.val, args.device)
        print(f"before: recorded / predicted length on {len(r)} val utterances: median {np.median(r):.3f}, "
              f"mean {r.mean():.3f}", flush=True)
    opt = torch.optim.AdamW(dp.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(min(200, args.steps // 10), args.steps, "cosine"))
    it, t0, agg = infinite(loader), time.time(), []
    for step in range(1, args.steps + 1):
        dur_in, x_mask, frames = duration_targets(model, next(it), args.device)
        loss = duration_fit_loss(dp(dur_in, x_mask), frames, x_mask, args.loss, edges=args.edges == "log")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dp.parameters(), 1.0)
        opt.step()
        sched.step()
        agg.append(float(loss.detach()))
        if step % args.log_every == 0 or step == args.steps:
            print(f"step {step} loss={np.mean(agg):.4f} ({len(agg) / (time.time() - t0):.1f} it/s)", flush=True)
            agg, t0 = [], time.time()
    dp.eval()
    if val is not None:
        r = length_ratios(model, val, args.val, args.device)
        print(f"after: recorded / predicted length on {len(r)} val utterances: median {np.median(r):.3f}, "
              f"mean {r.mean():.3f}", flush=True)
    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    new = {f"encoder.duration.{k}": v.detach().cpu() for k, v in dp.state_dict().items()}
    missing = set(new) - set(ck["ema"])
    if missing:
        raise KeyError(f"{args.model} has no {sorted(missing)[:3]}: not the model that was trained")
    ck["ema"] = {**ck["ema"], **{k: v.to(ck["ema"][k].dtype) for k, v in new.items()}}
    dropped = [k for k in ("duration_scale", "duration_scales") if ck.pop(k, None) is not None]
    ck["durations"] = {"loss": args.loss, "edges": args.edges, "steps": args.steps, "lr": args.lr,
                       "data": Path(root).name}
    save_checkpoint(Path(args.out), **ck)
    note = f"; dropped {', '.join(dropped)} (re-run calibrate-durations if wanted)" if dropped else ""
    print(f"wrote {args.out}{note}", flush=True)
