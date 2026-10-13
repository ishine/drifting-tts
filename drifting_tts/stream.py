"""Training batches streamed from parquet shards (local paths or ``hf://``) instead of a prepared ``mels.bin``, for a
corpus larger than the disk (VoiceHub/echo-clones-10kh-en: ~10,000 h, ~675 GB as log-mels).

Every DataLoader worker takes every n-th shard of a shuffled order and downloads the next one while it reads the
current one, so ``workers`` shards download at once and at most two per worker are on disk. Each row becomes a log-mel
exactly as ``prepare`` makes it (decode, trim, loudness, 24 kHz, the backend's front end) and its text goes through the
language's frontend; a shard's utterances are bucketed by length into batches of :func:`drifting_tts.data.collate`'s
format. The mel statistics, the speaker table, the language and the val split come from a prepared dataset of the same
corpus (``data.root``), so a streamed model is used like any other. Config::

    data:
      root: data/en_pilot        # stats.json, speakers.json, the val split
      stream:
        files: hf://datasets/VoiceHub/echo-clones-10kh-en/data/*.parquet
        speaker_column: speaker_id
        workers: 10
        cache: /workspace/stream_cache
"""

from __future__ import annotations

import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, IterableDataset

from .audio import make_logmel, prepare_waveform
from .data import MelDataset, MelStats, collate
from .text import frontend


def _hub_path(f: str) -> tuple[str, str]:
    """``hf://datasets/<owner>/<name>/<path>`` -> (repo id, path in the repo)."""
    parts = f.removeprefix("hf://").split("/")
    if parts[0] != "datasets" or len(parts) < 4:
        raise ValueError(f"expected hf://datasets/<owner>/<name>/<file>, got {f}")
    return f"{parts[1]}/{parts[2]}", "/".join(parts[3:])


class ParquetStream(IterableDataset):
    """Endless length-bucketed batches of the utterances of ``files`` (see the module docstring)."""

    def __init__(self, files: list[str], root: str | Path, speaker_column: str = "speaker", text_column: str = "text",
                 min_frames: int = 64, max_frames: int = 1500, batch_frames: int = 24000, max_batch: int = 24,
                 seed: int = 0, cache: str | Path | None = None, trim: bool = True):
        root = Path(root)
        st = json.loads((root / "stats.json").read_text())
        self.backend, self.language = st.get("backend", "vocos"), st.get("language", "tr")
        self.stats = MelStats.load(root)
        self.speakers = json.loads((root / "speakers.json").read_text())
        self.files = sorted(files)
        if not self.files:
            raise FileNotFoundError("no parquet files to stream")
        self.speaker_column, self.text_column = speaker_column, text_column
        self.min_frames, self.max_frames, self.batch_frames, self.max_batch = min_frames, max_frames, batch_frames, \
            max_batch
        self.seed, self.trim, self.epoch = seed, trim, 0
        self.cache = Path(cache) if cache else root / "stream_cache"

    def _shards(self, worker: int, workers: int):
        epoch = self.epoch
        while True:
            order = self.files[:]
            random.Random(self.seed + 1000 * epoch).shuffle(order)
            yield from order[worker::workers]
            epoch += 1

    def _fetch(self, f: str, worker: int) -> tuple[str | None, bool]:
        """A local path of shard ``f`` (downloaded for ``hf://``; ``True``: delete it after use)."""
        if not f.startswith("hf://"):
            return f, False
        from huggingface_hub import hf_hub_download

        repo, path = _hub_path(f)
        for attempt in range(5):
            try:
                return hf_hub_download(repo, path, repo_type="dataset", local_dir=self.cache / f"w{worker}"), True
            except Exception as e:  # network: retry, then skip the shard
                print(f"[stream] {f}: {type(e).__name__}: {e} (attempt {attempt + 1})", flush=True)
                time.sleep(10 * (attempt + 1))
        return None, False

    def _items(self, path: str, mel_fn, text) -> list[dict]:
        pf = pq.ParquetFile(path)
        cols = ["audio", self.text_column, self.speaker_column]
        items = []
        for rg in range(pf.num_row_groups):
            for row in pf.read_row_group(rg, columns=cols).to_pylist():
                spk = self.speakers.get(row[self.speaker_column])
                norm = text.normalize(row[self.text_column] or "")
                if spk is None or len(norm) < 2:
                    continue
                try:
                    wav = prepare_waveform(row["audio"]["bytes"], trim=self.trim)
                except Exception:  # corrupted audio
                    continue
                mel = mel_fn(wav)  # [n_mels, T]
                ids = text.text_to_ids(norm, normalized=True)
                if not torch.isfinite(mel).all() or not (self.min_frames <= mel.shape[1] <= self.max_frames) \
                        or len(ids) > mel.shape[1]:
                    continue
                items.append({"text": torch.tensor(ids, dtype=torch.long), "mel": self.stats.normalize(mel),
                              "spk": spk, "index": -1})
        return items

    def _batches(self, items: list[dict], rng: random.Random) -> list[list[dict]]:
        """A shard's utterances in batches of similar length within the frame budget (as BucketBatchSampler)."""
        items.sort(key=lambda x: x["mel"].shape[1])
        batches, cur = [], []
        for it in items:
            if cur and (len(cur) + 1 > self.max_batch or (len(cur) + 1) * it["mel"].shape[1] > self.batch_frames):
                batches.append(cur)
                cur = []
            cur.append(it)
        rng.shuffle(batches)
        return batches

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        worker, workers = (info.id, info.num_workers) if info else (0, 1)
        torch.set_num_threads(1)
        mel_fn, text = make_logmel(self.backend), frontend(self.language)
        rng = random.Random(self.seed + 7919 * worker + self.epoch)
        shards = self._shards(worker, workers)
        with ThreadPoolExecutor(1) as pool:
            ahead = pool.submit(self._fetch, next(shards), worker)
            while True:
                path, temporary = ahead.result()
                ahead = pool.submit(self._fetch, next(shards), worker)
                if path is None:
                    continue
                items = self._items(path, mel_fn, text)
                if temporary:
                    os.remove(path)
                for batch in self._batches(items, rng):
                    yield collate(batch)


def stream_files(pattern: str) -> list[str]:
    from .prepare import _glob

    return _glob(pattern)


def stream_loader(cfg) -> tuple[MelDataset, DataLoader]:
    """``data.stream`` of a config: the val split of ``data.root`` (stats, speakers, language of the streamed model)
    and an endless DataLoader of streamed batches."""
    d, t, s = cfg.data, cfg.train, cfg.data.stream
    ref = MelDataset(d.root, "val", min_frames=1, max_frames=10**9)
    files = stream_files(s.files)
    stream = ParquetStream(files, d.root, speaker_column=s.get("speaker_column", "speaker"),
                           text_column=s.get("text_column", "text"), min_frames=cfg.drift.crop_frames,
                           max_frames=d.max_frames, batch_frames=t.batch_frames, max_batch=t.batch_size, seed=t.seed,
                           cache=s.get("cache"))
    workers = int(s.get("workers", 8))
    loader = DataLoader(stream, batch_size=None, num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
                        prefetch_factor=4 if workers > 0 else None)
    print(f"stream: {len(files)} shards ({s.files}), {workers} workers", flush=True)
    return ref, loader
