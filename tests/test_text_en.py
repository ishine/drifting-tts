"""English text frontend (``language: en``): normalisation, symbols, and the language switch of the pipeline."""

import pytest

from drifting_tts import text_en
from drifting_tts.text import BLANK_ID, PAD_ID, frontend

CASES = [
    ("Uh, you know, it's— well, I got maybe four hours?", "uh, you know, it's, well, i got maybe four hours?"),
    ("I left at 7:45 to beat the rush.", "i left at seven forty five to beat the rush."),
    ("Sleeping at 4am is great.", "sleeping at four ay em is great."),
    ("Meet me at 10:00 or 9:05 p.m.", "meet me at ten o'clock or nine oh five pee em."),
    ("You ever lay there at three AM?", "you ever lay there at three ay em?"),
    ("Those were the 90s, the 1980's.", "those were the nineties, the nineteen eighties."),
    ("Since 2008, in 1999, by 2024, in 1900.",
     "since two thousand eight, in nineteen ninety nine, by twenty twenty four, in nineteen hundred."),
    ("1,250,000 views and 3.14", "one million two hundred fifty thousand views and three point one four"),
    ("It's $15, or $4.99, or $1, or $2.5 million.",
     "it's fifteen dollars, or four dollars ninety nine cents, or one dollar, or two point five million dollars."),
    ("Like 20% off.", "like twenty percent off."),
    ("The 21st or the 2nd, the 40th.", "the twenty first or the second, the fortieth."),
    ("His BBQ, an old TV, a call from HR.", "his barbecue, an old tee vee, a call from aitch ar."),
    ("The FBI and NASA, in the USA.", "the eff bee eye and nasa, in the you ess ay."),
    ("I was SO tired, hmm, HMM. OK?", "i was so tired, hmm, hmm. okay?"),
    ("Mr. Smith vs. Dr. Jones, etc.", "mister smith versus doctor jones, et cetera."),
    ("It ends at 5 p.m. Then we go, e.g. home.", "it ends at five pee em. then we go, for example home."),
    ("It's -5 degrees, pages 20-30, No. 7.", "it's minus five degrees, pages twenty to thirty, number seven."),
    ("We'll see, rock'n'roll, the students' books, 'quoted'.",
     "we'll see, rock'n'roll, the students books, quoted."),
    ("café naïve “quoted” & more", "cafe naive quoted and more"),
]


@pytest.mark.parametrize("raw,expected", CASES)
def test_normalize(raw, expected):
    assert text_en.normalize(raw) == expected


def test_numbers():
    assert text_en.number_to_words(0) == "zero"
    assert text_en.number_to_words(1001) == "one thousand one"
    assert text_en.number_to_words(-12) == "minus twelve"
    assert text_en.ordinal_to_words(22) == "twenty second"
    assert text_en.ordinal_to_words(12) == "twelfth"
    assert text_en.year_to_words(2015) == "twenty fifteen"
    assert text_en.year_to_words(1905) == "nineteen oh five"


def test_symbols_and_frontend_switch():
    en, tr = frontend("en"), frontend("tr")
    assert en is text_en and tr.SYMBOLS[:2] == en.SYMBOLS[:2]
    assert (en.PAD_ID, en.BLANK_ID) == (PAD_ID, BLANK_ID)
    assert "'" in en.SYMBOLS and "ç" not in en.SYMBOLS
    ids = en.text_to_ids("Don't stop!")
    assert ids[0::2] == [BLANK_ID] * (len(ids) // 2 + 1) and en.ids_to_text(ids) == "don't stop!"
    assert all(c in en.SYMBOL_TO_ID for c in en.normalize("Ünïcödé — 100% “ok” @ #3 ½"))
    with pytest.raises(ValueError):
        frontend("de")


def test_english_ar_model_vocabulary():
    from drifting_tts.config import load_config
    from drifting_tts.models.ar import ARDriftTTS

    cfg = load_config("configs/ar_tts.yaml", ["model.text.language=en", "model.ar.depth=1", "model.ar.head_depth=1"])
    model = ARDriftTTS(cfg.model, num_speakers=2)
    assert model.encoder.emb.num_embeddings == len(text_en.SYMBOLS)


def _english_parquet(path, n, start=0, seconds=1.0):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from tests.test_data import _wav_bytes

    rows = {
        "audio": [{"bytes": _wav_bytes(seconds + 0.1 * i, freq=200 + 10 * i), "path": f"{i}.wav"} for i in range(n)],
        "text": [f"Hello there, it's {start + i} o'clock." for i in range(n)],
        "speaker_id": [f"spk_{i % 3}" for i in range(n)],
    }
    pq.write_table(pa.table(rows), path)


def test_prepare_english_held_out_files_and_stream(tmp_path):
    import json

    import torch

    from drifting_tts.cli import main
    from drifting_tts.data import MelDataset
    from drifting_tts.stream import ParquetStream

    _english_parquet(tmp_path / "train-0.parquet", 12)
    _english_parquet(tmp_path / "held-0.parquet", 6, start=100, seconds=2.5)
    out = tmp_path / "prep"
    main(["prepare", "--parquet-glob", str(tmp_path / "*.parquet"), "--val-glob", str(tmp_path / "held-*.parquet"),
          "--val-size", "4", "--out", str(out), "--workers", "1", "--no-trim", "--backend", "bigvgan",
          "--language", "en", "--speaker-column", "speaker_id"])
    stats = json.loads((out / "stats.json").read_text())
    assert stats["language"] == "en" and stats["speakers"] == 3
    index = [json.loads(line) for line in open(out / "index.jsonl")]
    assert index[0]["norm_text"] == "hello there, it's zero o'clock." and index[0]["speaker"] == "spk_0"
    splits = [e["split"] for e in index]
    assert splits[:12] == ["train"] * 12 and sorted(splits[12:]) == ["dev"] * 2 + ["val"] * 4  # held-out file only
    ds = MelDataset(out, "train", min_frames=1)
    assert ds.language == "en" and text_en.ids_to_text(ds[0]["text"].tolist()) == index[0]["norm_text"]

    stream = ParquetStream([str(tmp_path / "train-0.parquet")], out, speaker_column="speaker_id", min_frames=1,
                           batch_frames=10**6, max_batch=4, trim=False)
    batch = next(iter(stream))
    assert sorted(batch) == ["index", "mel", "mel_len", "spk", "text", "text_len"]
    assert batch["mel"].shape[:2] == (4, 100) and batch["text"].shape[0] == 4
    k = int(batch["mel_len"].argmin())  # the streamed mel equals the prepared one (fp16 storage aside)
    i = next(j for j, e in enumerate(index) if e["frames"] == int(batch["mel_len"][k]) and e["split"] == "train")
    torch.testing.assert_close(batch["mel"][k, :, : index[i]["frames"]], ds[i]["mel"], atol=2e-3, rtol=0)
