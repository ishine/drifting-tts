"""Prepare the files of the English release for upload to Vyvo/drifting-tts-en: export, verify, scan (no upload).

Writes ``<out>/ardrift_en.pt`` (the acoustic model, ``drifting_tts.publish.export_tts``, fp32), ``<out>/bigvgan_en.pt``
(the fine-tuned BigVGAN-v2, ``export_bigvgan``, fp32), ``<out>/voices.json`` and ``<out>/README.md`` (the model card,
both from ``release_en/``). Each checkpoint is reloaded on the CPU and checked against its source: the same weights,
and for the acoustic model the same mel for two sentences at a fixed seed. Then every staged file is scanned for
absolute paths and for the data root's names (the data root's own name, the speaker names). The script exits with an
error if anything is found.

    python scripts/prepare_english_release.py --model runs/en_dur/model_poisson.pt \
        --vocoder runs/en_bigvgan_gta/bigvgan_ft_40000.pt --data data/en_pilot --out runs/en_release
    hf upload Vyvo/drifting-tts-en runs/en_release .
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch

from drifting_tts.publish import export_bigvgan, export_tts, private_terms, scan

TEXTS = ["Hello! It's 7:45, and the meeting moved to the 21st.", "I spent $12.50 on coffee, about 20% of my budget."]
HERE = Path(__file__).resolve().parent.parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_tts(src: str, dst: Path, speakers: list[int]) -> None:
    from drifting_tts.text import frontend
    from drifting_tts.train import load_tts

    (a, cfg, _), (b, _, _) = load_tts(src), load_tts(dst)
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa), "weights differ"
    fe = frontend(cfg.model.text.get("language"))
    for text in TEXTS:
        ids = torch.tensor([fe.text_to_ids(fe.normalize(text), normalized=True)])
        for spk in speakers[:2]:
            out = [m.synthesize(ids, torch.tensor([ids.shape[1]]), torch.tensor([spk]), temperature=0.5,
                                generator=torch.Generator().manual_seed(0))[0] for m in (a, b)]
            assert torch.equal(*out), f"mel differs: {text!r}, speaker {spk}"


def check_vocoder(src: str, dst: Path) -> None:
    a, b = (torch.load(p, map_location="cpu", weights_only=False) for p in (src, dst))
    assert all(torch.equal(a["generator"][k], b["generator"][k]) for k in a["generator"]), "generator differs"
    assert b.keys() <= {"generator", "hparams", "repo", "step", "noise_channels"}, "training state left in"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True, help="the English model with its re-fitted durations (train-durations)")
    p.add_argument("--vocoder", required=True, help="the BigVGAN-v2 fine-tune (finetune-vocoder bigvgan_ft_<step>.pt)")
    p.add_argument("--data", required=True, help="the prepared data root whose names must not appear (data/en_pilot)")
    p.add_argument("--out", required=True, help="staging directory, laid out like the Hub repo")
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    voices = json.loads((HERE / "release_en" / "voices.json").read_text())["voices"]

    torch.save(export_tts(torch.load(args.model, map_location="cpu", weights_only=False)), out / "ardrift_en.pt")
    check_tts(args.model, out / "ardrift_en.pt", [v["id"] for v in voices])
    torch.save(export_bigvgan(torch.load(args.vocoder, map_location="cpu", weights_only=False)), out / "bigvgan_en.pt")
    check_vocoder(args.vocoder, out / "bigvgan_en.pt")
    shutil.copyfile(HERE / "release_en" / "voices.json", out / "voices.json")
    shutil.copyfile(HERE / "release_en" / "README.md", out / "README.md")
    print("ardrift_en.pt and bigvgan_en.pt verified against their sources")

    terms, ok = private_terms(args.data), True
    print("scan (hits per category; 0 = clean):")
    for name in ("ardrift_en.pt", "bigvgan_en.pt"):
        hits = scan(out / name, terms)
        ok &= not any(hits.values())
        print(f"  {name}: {(out / name).stat().st_size / 2**20:.1f} MB, sha256 {sha256(out / name)[:16]}..., "
              + ", ".join(f"{k} {v}" for k, v in hits.items()))
    for name in ("voices.json", "README.md"):
        text = (out / name).read_text().lower()
        hits = {cat: sum(text.count(w.lower()) for w in set(words) if len(w) >= 4) for cat, words in terms.items()}
        ok &= not any(hits.values())
        print(f"  {name}: " + ", ".join(f"{k} {v}" for k, v in hits.items()))
    if not ok:
        sys.exit("a staged file contains private strings: do not upload it")


if __name__ == "__main__":
    main()
