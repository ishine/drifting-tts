# ARDrift in English: 10,000 hours of synthetic speech

ARDrift ([docs/AR_DRIFT.md](AR_DRIFT.md)) trained in English from scratch, with an English character frontend, on 10,000
hours of synthetic speech too large for the disk, so streamed from the Hub while training.

- [Data](#data)
- [What changed in the code](#what-changed-in-the-code)
- [Results](#results)
- [Limits](#limits)
- [Commands](#commands)

## Data

[VoiceHub/echo-clones-10kh-en](https://huggingface.co/datasets/VoiceHub/echo-clones-10kh-en): the first 6,949 parquet
shards, unchanged, of [SynDataLab-EN/echo-clones-4m-en](https://huggingface.co/datasets/SynDataLab-EN/echo-clones-4m-en):

| | |
|---|---|
| hours / utterances | 10,000.7 h / 3,474,250 (1.33 TB) |
| speech | synthetic, generated with [EchoTTS (jordand/echo-tts-base)](https://huggingface.co/jordand/echo-tts-base) for 4,000 reference speakers, 44.1 kHz |
| text | conversational English, 24-420 characters (fillers, em dashes, few digits or acronyms) |
| license | Apache-2.0 as declared by the source; the generating model is CC-BY-NC-SA-4.0 |

The source's shards are not uniform: the first ones hold mid-length utterances (median 8.3 s in shards 0-69), later
ones many short ones (median 5.6 s in shard 3,000).

**Held out:** the val and dev splits (200 + 300 utterances) come from source shard 07,960, which is not among the
6,949 shards: texts never trained on, read by the training speakers (every speaker occurs throughout the corpus).

## What changed in the code

- **English frontend** (`drifting_tts/text_en.py`): the 26 letters, the apostrophe (`we'll`, `don't`: it changes the
  pronunciation) and the Turkish frontend's punctuation, with `<pad>` / `<blank>` at the same ids. Normalisation
  writes out numbers, years (`1999` → *nineteen ninety nine*), decades, ordinals, money, percentages, clock times,
  common abbreviations and all-caps acronyms letter by letter (`TV` → *tee vee*, while emphasis like `SO` stays a word).
  English spelling is far from phonemic; with this much data the text encoder learns the letters' sounds (no G2P).
- **A language switch**: `model.text.language` sets the text encoder's vocabulary and a prepared dataset's
  `stats.json` records its language (`prepare --language`); the Synthesizer, `evaluate` and Whisper follow them. A
  model or dataset without a language is Turkish, as before.
- **`prepare`**: `--speaker-column`, `--val-glob` (held-out files form the val split, the rest of them dev), and
  `hf://` paths read without a local copy.
- **Streaming** (`drifting_tts/stream.py`, `data.stream`): 10,000 h of log-mels would take ~675 GB. Each DataLoader
  worker downloads every n-th shard of a shuffled order (the next one while it reads the current one), turns each row
  into a log-mel as `prepare` does and yields length-bucketed batches; at most two shards per worker are on disk. One
  core makes mels ~1,400x faster than real time, so the network decides: 12 workers deliver ~2,165 s of audio per
  second, more than the training uses. The mel statistics, the speaker table and the val split come from the
  prepared pilot data.
- `evaluate --asr-device cpu`: Whisper beside a training run that holds the GPU.

## Results

Val 100 (the first 100 held-out utterances): Whisper large-v3 in English (for the 10k-h run on the CPU, int8),
UTMOSv2, WavLM-ECAPA similarity to the recording's copy-synthesis; Vocos v2, noise temperature 0.5, no guidance. The
copy-synthesis row is the recording's mel through Vocos v2: the ceiling of this vocoder.

**Pilot** (`configs/ar_tts_en.yaml`: 87.7 h, source shards 0-69; the Turkish model's size with a wider text encoder:
12.8 + 42.5 + 18.0 M; 30k steps, 7.2 it/s):

| step | WER | CER | UTMOSv2 | SIM |
|---|---|---|---|---|
| 10k | 22.82% | 12.61% | 2.742 | 0.190 |
| 20k | 4.74% | 2.39% | 2.912 | 0.329 |
| 30k | 3.32% | 1.52% | 2.957 | 0.422 |
| copy-synthesis | 2.57% | 1.12% | 2.926 | – |

**10,000 h** (`configs/ar_tts_en_10k.yaml`: 36.2 + 121.2 + 20.5 M, batches of up to 48 utterances, 200k steps, about
one pass over the data, 13 h on one RTX 5090):

| step | WER | CER | UTMOSv2 | SIM |
|---|---|---|---|---|
| 20k | 3.99% | 2.04% | 2.917 | 0.394 |
| 40k | 2.51% | 1.22% | 2.950 | 0.591 |
| 60k | 2.51% | 1.26% | 3.014 | 0.653 |
| 80k | 2.30% | 1.03% | 2.907 | 0.695 |
| 100k | 2.17% | 0.94% | 2.945 | 0.722 |
| 120k | 2.51% | 1.07% | 2.990 | 0.740 |
| 140k | 2.10% | 0.98% | 2.904 | 0.757 |
| 160k | 2.57% | 1.07% | 2.998 | 0.771 |
| 180k | 2.10% | 0.92% | 2.949 | 0.770 |
| 200k | 2.17% | 0.90% | 2.994 | 0.774 |
| copy-synthesis | 2.44% | 1.09% | 2.926 | – |

**All 500 held-out utterances** (val 200 + dev 300, 15,302 words; Whisper on the GPU, fp16), to tell the last
snapshots apart:

| step | WER | CER | UTMOSv2 | SIM |
|---|---|---|---|---|
| 140k | 1.50% | 0.65% | 2.916 | 0.783 |
| 180k | **1.39%** | **0.62%** | 2.921 | 0.800 |
| **200k** (the model in the test Space) | 1.42% | 0.64% | **2.930** | **0.803** |
| copy-synthesis | 1.57% | 0.73% | 2.907 | – |

180k and 200k differ by 5 words; 200k is ahead on similarity and UTMOSv2.

- **As intelligible as the recordings from 40k steps on**, with the vocoder they go through, and slightly more on
  all 500 utterances (1.42% against 1.57%: the source's own slips are not learned); the 100 utterances hold ~1,500
  words, so the snapshots after 40k differ there by a few words.
- **The voices are what more data buys**: similarity to each speaker rises at every snapshot, 0.39 → 0.77, where the
  pilot, with ~9 utterances per speaker, stopped at 0.42.
- **UTMOSv2 at the recordings' level** (2.90-3.01 against 2.926): the synthetic source is the ceiling.

## Limits

- **Synthetic data**: the model learns EchoTTS's voices and its artefacts; a recorded corpus would be needed to go past
  them.
- **License**: the generating model's CC-BY-NC-SA-4.0 may restrict commercial use of a model trained on this data.
- **Speakers by ID only** (4,000 table rows), no cloning from a reference recording.
- **The test set is from the same synthetic corpus**: texts never seen, voices seen. There is no comparison with
  other English TTS systems on a common benchmark yet.

## Commands

```bash
# pilot data (also the mel statistics, speaker table and val split of the 10k-h run)
drifting-tts prepare --parquet-glob "hf://datasets/SynDataLab-EN/echo-clones-4m-en/data/train-000[0-6][0-9].parquet" \
    --val-glob "hf://datasets/SynDataLab-EN/echo-clones-4m-en/data/train-07960.parquet" --val-size 200 \
    --out data/en_pilot --backend bigvgan --language en --speaker-column speaker_id --workers 24
drifting-tts train-ar --config configs/ar_tts_en.yaml --workdir runs/ar_en_pilot
drifting-tts train-ar --config configs/ar_tts_en_10k.yaml --workdir runs/ar_en_10k     # streamed from the Hub
drifting-tts evaluate --model runs/ar_en_10k/model_ema.pt --split val --num 100 --temperature 0.5 --vocoder vocos-v2
```

```python
tts = Synthesizer("runs/ar_en_10k/model_ema.pt", "cuda", vocoder="vocos-v2")
wav, _ = tts("Hello! It's 7:45, and the 90s called.", speaker="spk_0042", temperature=0.5)
```
