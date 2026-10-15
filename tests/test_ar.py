"""ARDrift (#42): patching, causality, cached generation against teacher forcing, a training step, and the CLI from
training to synthesis. Tiny models on the CPU."""

import torch

from drifting_tts.config import Config, load_config
from drifting_tts.models.ar import ARDriftTTS, ar_features, patchify, unpatchify

CFG = {"kind": "ar", "text": {"d": 16, "heads": 2, "layers": 1, "ffn": 32, "dropout": 0.0, "spk_dim": 8},
       "ar": {"patch": 4, "lookahead": 2, "dim": 32, "depth": 2, "heads": 2, "head_dim": 32, "head_depth": 2}}


def _model(seed: int = 0) -> ARDriftTTS:
    torch.manual_seed(seed)
    m = ARDriftTTS(Config(CFG), num_speakers=3, n_mels=100).eval()
    with torch.no_grad():
        for p in m.parameters():  # zero-initialised output layers: make every path visible
            p.add_(torch.randn_like(p) * 0.05)
    return m


def test_patchify_roundtrip():
    x = torch.randn(2, 100, 12)
    p = patchify(x, 4)
    assert p.shape == (2, 3, 400) and torch.equal(p[0, 1, :100], x[0, :, 4])  # frame-major inside a patch
    assert torch.equal(unpatchify(p, 4, 100), x)


def test_backbone_is_causal_in_the_mel_and_sees_the_text_lookahead():
    m = _model()
    cond, y, spk = torch.randn(1, 116, 32), torch.randn(1, 100, 32), torch.tensor([1])
    s = m.teacher_forced(cond, y, spk)
    y2 = y.clone()
    y2[..., 20:] += 1.0  # changes patches 5.. : the inputs of steps 6..
    s2 = m.teacher_forced(cond, y2, spk)
    assert torch.allclose(s[:, :6], s2[:, :6], atol=1e-5) and not torch.allclose(s[:, 6:], s2[:, 6:])
    c2 = cond.clone()
    c2[..., 16:20] += 1.0  # the text of step 4: seen by steps 2-4 (lookahead 2), not by 0-1
    s3 = m.teacher_forced(c2, y, spk)
    assert torch.allclose(s[:, :2], s3[:, :2], atol=1e-5) and not torch.allclose(s[:, 2:5], s3[:, 2:5])


def test_cached_generation_matches_teacher_forcing():
    m = _model()
    cond, spk = torch.randn(2, 116, 29), torch.tensor([0, 2])  # 29 frames: the last patch is padded
    mel = m.generate(cond, spk, temperature=0.0)
    assert mel.shape == (2, 100, 29)
    padded = m.pad_frames(cond)
    s = m.teacher_forced(padded, m.pad_frames(mel), spk)
    z = torch.zeros(2 * s.shape[1], m.head.noise_dim)
    again = patchify(padded[:, :100], 4) + m.head(s.flatten(0, 1), z).view(2, s.shape[1], -1)
    torch.testing.assert_close(unpatchify(again, 4, 100)[..., :29], mel, atol=1e-4, rtol=1e-4)
    g = [torch.Generator().manual_seed(3) for _ in range(2)]
    assert torch.equal(m.generate(cond, spk, 0.7, g[0]), m.generate(cond, spk, 0.7, g[1]))


def test_features_and_training_step_backpropagate():
    from drifting_tts.train_ar import training_step

    m = _model().train()
    cfg = load_config("configs/ar_tts.yaml", ["drift.gen_per_cond=4", "drift.max_rows=8"])
    text = torch.randint(3, 30, (2, 11))
    batch = {"text": text, "text_len": torch.tensor([11, 9]), "mel": torch.randn(2, 100, 40),
             "mel_len": torch.tensor([40, 30]), "spk": torch.tensor([0, 1])}
    tau = torch.nn.Parameter(torch.tensor(1.0))
    loss, metrics = training_step(m, batch, cfg, "cpu", tau)
    loss.backward()
    assert torch.isfinite(loss) and metrics["rows"] == 8
    assert tau.grad is not None and m.head.out.weight.grad.abs().sum() > 0
    assert m.blocks[0].attn.qkv.weight.grad is not None
    f = ar_features(torch.randn(3, 5, 400), 4, 100)
    assert f["patch"].shape == (3, 5, 1, 400) and f["frame"].shape == (3, 5, 4, 100) == f["detail"].shape


def test_train_ar_cli_and_synthesis(tmp_path):
    from drifting_tts.cli import main
    from drifting_tts.synthesize import Synthesizer
    from tests.test_data import _fake_parquet

    _fake_parquet(tmp_path / "x.parquet", n=12)
    data = tmp_path / "prep"
    main(["prepare", "--parquet-glob", str(tmp_path / "*.parquet"), "--out", str(data), "--workers", "1",
          "--val-size", "2", "--no-trim", "--backend", "bigvgan"])
    work = tmp_path / "run"
    main(["train-ar", "--workdir", str(work), f"data.root={data}", "data.filters={}", "train.cpu=true",
          "train.steps=2", "train.batch_size=3", "train.num_workers=0", "train.log_every=1", "train.save_every=2",
          "train.sample_every=0", "train.warmup=1", "drift.gen_per_cond=3", "drift.max_rows=16",
          "model.ar.dim=32", "model.ar.depth=1", "model.ar.heads=2", "model.ar.head_dim=32", "model.ar.head_depth=1",
          "model.text.d=16", "model.text.layers=1", "model.text.ffn=32", "model.text.spk_dim=8"])
    synth = Synthesizer(work / "model_ema.pt", "cpu", vocoder="griffin-lim")
    synth.vocoder.model.n_iter = 2
    wav, info = synth("merhaba dünya. nasılsın?", speaker=0, temperature=0.5, seed=1)
    assert wav.numel() > 0 and torch.isfinite(wav).all()
    assert torch.equal(synth("merhaba dünya. nasılsın?", speaker=0, temperature=0.5, seed=1)[0], wav)
    streamed = torch.cat(list(synth.stream("merhaba dünya. nasılsın?", speaker=0, temperature=0.5, seed=1)))
    assert streamed.shape == wav.shape  # Griffin-Lim does not stream: one piece per sentence, the same draws
    torch.testing.assert_close(streamed, wav, atol=1e-4, rtol=1e-4)


def test_fast_generator_matches_generate():
    import pytest

    from drifting_tts.models.ar import ARGenerator

    m = _model()
    cond, spk = torch.randn(2, 116, 29), torch.tensor([0, 2])
    ref = m.generate(cond, spk, 0.7, torch.Generator().manual_seed(5))
    fast = ARGenerator(m, max_steps=16, graphs=False)
    torch.testing.assert_close(fast(cond, spk, 0.7, torch.Generator().manual_seed(5)), ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(fast(cond, spk, 0.7, torch.Generator().manual_seed(5)), ref, atol=1e-4, rtol=1e-4)
    if not torch.cuda.is_available():
        pytest.skip("CUDA graphs need a GPU")
    mc = m.cuda()
    ref = mc.generate(cond.cuda(), spk.cuda(), 0.7, torch.Generator(device="cuda").manual_seed(5))
    g = ARGenerator(mc, max_steps=16)
    for _ in range(2):  # the second call replays the captured graph on the stale caches of the first
        out = g(cond.cuda(), spk.cuda(), 0.7, torch.Generator(device="cuda").manual_seed(5))
        torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


def test_guidance_on_the_backbone_state():
    from drifting_tts.models.ar import ARGenerator

    m = _model()
    cond, spk = torch.randn(2, 116, 21), torch.tensor([0, 2])
    plain = m.generate(cond, spk, 0.5, torch.Generator().manual_seed(1))
    torch.testing.assert_close(m.generate(cond, spk, 0.5, torch.Generator().manual_seed(1), cfg_scale=1.0), plain)
    guided = m.generate(cond, spk, 0.5, torch.Generator().manual_seed(1), cfg_scale=2.0)
    assert not torch.allclose(guided, plain)
    fast = ARGenerator(m, max_steps=8, graphs=False)
    torch.testing.assert_close(fast(cond, spk, 0.5, torch.Generator().manual_seed(1), 2.0), guided, atol=1e-4,
                               rtol=1e-4)
    y = torch.randn(2, 100, 24)
    dropped = m.teacher_forced(m.pad_frames(cond), y, spk, drop=torch.tensor([True, False]))
    null = m.teacher_forced(torch.zeros_like(m.pad_frames(cond)), y, spk)
    full = m.teacher_forced(m.pad_frames(cond), y, spk)
    torch.testing.assert_close(dropped[0], null[0])
    torch.testing.assert_close(dropped[1], full[1])


def test_gta_draws_each_patch_after_the_recorded_ones():
    """GTA (finetune-vocoder): patch t is the head's draw from the state after the *recorded* patches 0..t-1, i.e. what
    generation continuing a prefix of the recording draws first; at temperature 0 the two agree exactly."""
    m = _model()
    cond, y, spk = torch.randn(1, 116, 32), torch.randn(1, 100, 32), torch.tensor([1])
    g = m.gta(cond, y, spk, temperature=0.0)
    assert g.shape == y.shape
    for t in (0, 3, 6):
        nxt = m.generate(cond, spk, temperature=0.0, prefix=y[..., : 4 * t] if t else None)[..., 4 * t: 4 * t + 4]
        torch.testing.assert_close(g[..., 4 * t: 4 * t + 4], nxt, atol=1e-5, rtol=1e-4)
    gen = lambda: torch.Generator().manual_seed(3)  # noqa: E731
    a, b = m.gta(cond, y, spk, 0.5, gen()), m.gta(cond, y, spk, 0.5, gen())
    assert torch.equal(a, b) and not torch.allclose(a, g)
    assert m.gta(cond, y[..., :30], spk, 0.5).shape == (1, 100, 30)  # not a whole number of patches


def test_band_energy_follows_the_loudest_bins():
    from drifting_tts.models.ar import band_energy

    quiet = torch.full((2, 100), -3.0)
    loud = quiet.clone()
    loud[:, 10] = 1.0  # one loud bin in band 0
    e_q, e_l = band_energy(quiet, 4.0, (0, 50, 100)), band_energy(loud, 4.0, (0, 50, 100))
    assert e_q.shape == (2, 2)
    torch.testing.assert_close(e_q, torch.full((2, 2), -3.0))
    # the log-mel mean of band 0 moves by 4 / 50 = 0.08; its energy moves to within log(50) / 4 of the loud bin
    assert e_l[0, 0] > 0.0 and torch.equal(e_l[:, 1], e_q[:, 1])
    f = ar_features(torch.randn(3, 5, 400), 4, 100, energy_bands=(0, 25, 50, 75, 100), energy_scale=4.0)
    assert f["energy"].shape == (3, 5, 4, 4)


def test_training_step_with_band_energies_and_own_history():
    import pytest

    from drifting_tts.train_ar import training_step

    m = _model().train()
    cfg = load_config("configs/ar_tts.yaml", ["drift.gen_per_cond=4", "drift.max_rows=8",
                                              "drift.energy_bands=[0,25,50,75,100]", "drift.self_history=1.0"])
    batch = {"text": torch.randint(3, 30, (2, 11)), "text_len": torch.tensor([11, 9]),
             "mel": torch.randn(2, 100, 40), "mel_len": torch.tensor([40, 30]), "spk": torch.tensor([0, 1])}
    tau = torch.nn.Parameter(torch.tensor(1.0))
    with pytest.raises(ValueError, match="energy_bands"):
        training_step(m, batch, cfg, "cpu", tau)
    loss, metrics = training_step(m, batch, cfg, "cpu", tau, mel_std=2.4)
    loss.backward()
    assert torch.isfinite(loss) and "p_data" in metrics
    assert m.head.out.weight.grad.abs().sum() > 0 and m.blocks[0].attn.qkv.weight.grad is not None

    # self_history: the backbone sees the head's own one-step patches as the history, not the recording
    seen = []
    orig = m.teacher_forced
    m.teacher_forced = lambda cond, y, *a: seen.append(y) or orig(cond, y, *a)
    training_step(m, batch, cfg, "cpu", tau, mel_std=2.4)
    hist = seen[-1]
    assert hist.shape[-1] == 40 and not torch.allclose(hist, batch["mel"][:, :, :40])
