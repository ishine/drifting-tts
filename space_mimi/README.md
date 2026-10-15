---
title: Drifting TTS English on Mimi (private test)
emoji: 🎙️
colorFrom: indigo
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
license: other
short_description: ARDrift English on Kyutai's Mimi latents (private test)
---

Private test of an unreleased experiment of [kadirnar/drifting-tts](https://github.com/kadirnar/drifting-tts): the
English ARDrift model (10,000 h of synthetic speech) trained on the latents of Mimi, the codec of Kyutai's Pocket TTS,
instead of log-mels (docs/MIMI.md). It is decoded with Kyutai's released Mimi decoder
([kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts), CC BY 4.0; code: the `pocket-tts` package, MIT).
Not a release.
