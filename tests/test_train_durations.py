"""train-durations (#72): the Poisson fit predicts the mean duration, and the command changes only the duration
predictor of the checkpoint it copies. Tiny models on the CPU."""

import torch

from drifting_tts.train_durations import duration_fit_loss


def test_poisson_aims_half_a_frame_below_the_mean_and_log_at_the_geometric_mean():
    # one token read with 1, 1 and 40 frames (a comma without and with a pause): mean 14, geometric mean ~3.4; the
    # Poisson fit aims half a frame lower, since synthesis rounds up
    frames = torch.tensor([[1.0], [1.0], [40.0]])
    mask = torch.ones(3, 1, 1)
    best = {}
    for kind in ("poisson", "log"):
        logw = torch.zeros(1, 1, 1, requires_grad=True)
        opt = torch.optim.Adam([logw], lr=0.05)
        for _ in range(2000):
            opt.zero_grad()
            duration_fit_loss(logw.expand(3, 1, 1), frames, mask, kind).backward()
            opt.step()
        best[kind] = float(logw.detach().exp())
    assert abs(best["poisson"] - 13.5) < 0.1
    assert abs(best["log"] - 40 ** (1 / 3)) < 0.05


def test_padding_is_ignored():
    logw = torch.zeros(1, 1, 3)
    mask = torch.tensor([[[1.0, 1.0, 0.0]]])
    a = duration_fit_loss(logw, torch.tensor([[2.0, 3.0, 0.0]]), mask, "poisson")
    b = duration_fit_loss(logw, torch.tensor([[2.0, 3.0, 99.0]]), mask, "poisson")
    assert torch.equal(a, b)


def test_train_durations_cli_changes_only_the_duration_predictor(tmp_path):
    from drifting_tts.cli import main
    from drifting_tts.synthesize import Synthesizer
    from tests.test_data import _fake_parquet

    _fake_parquet(tmp_path / "x.parquet", n=12)
    data = tmp_path / "prep"
    main(["prepare", "--parquet-glob", str(tmp_path / "*.parquet"), "--out", str(data), "--workers", "1",
          "--val-size", "2", "--no-trim", "--backend", "bigvgan"])
    work = tmp_path / "run"
    main(["train-ar", "--workdir", str(work), f"data.root={data}", "data.filters={}", "train.cpu=true",
          "train.steps=1", "train.batch_size=3", "train.num_workers=0", "train.log_every=1", "train.save_every=1",
          "train.sample_every=0", "train.warmup=1", "drift.gen_per_cond=3", "drift.max_rows=16",
          "model.ar.dim=32", "model.ar.depth=1", "model.ar.heads=2", "model.ar.head_dim=32", "model.ar.head_depth=1",
          "model.text.d=16", "model.text.layers=1", "model.text.ffn=32", "model.text.spk_dim=8"])
    src = torch.load(work / "model_ema.pt", weights_only=False)
    src["duration_scale"], src["duration_scales"] = 1.3, {0: 1.2}  # calibrated on the old predictor
    torch.save(src, work / "model_ema.pt")
    out = tmp_path / "durations.pt"
    main(["train-durations", "--model", str(work / "model_ema.pt"), "--out", str(out), "--steps", "3",
          "--batch-size", "3", "--num-workers", "0", "--val", "2", "--lr", "1e-2", "--device", "cpu"])
    new = torch.load(out, weights_only=False)
    changed = {k for k, v in new["ema"].items() if not torch.equal(v, src["ema"][k])}
    assert changed and all(k.startswith("encoder.duration.") for k in changed)
    assert new["ema"].keys() == src["ema"].keys()
    assert new["durations"]["loss"] == "poisson" and new["durations"]["steps"] == 3
    assert "duration_scale" not in new and "duration_scales" not in new
    from drifting_tts.publish import export_durations

    small = export_durations(new)  # what a Space that already holds the model gets
    assert small["durations"] == new["durations"]
    pred = Synthesizer(work / "model_ema.pt", "cpu", vocoder="griffin-lim").model.encoder.duration
    pred.load_state_dict(small["duration"])
    assert all(torch.equal(v, new["ema"][f"encoder.duration.{k}"]) for k, v in pred.state_dict().items())
    synth = Synthesizer(out, "cpu", vocoder="griffin-lim")
    synth.vocoder.model.n_iter = 2
    wav, _ = synth("merhaba dünya.", speaker=0, temperature=0.5, seed=1)
    assert wav.numel() > 0 and torch.isfinite(wav).all()
