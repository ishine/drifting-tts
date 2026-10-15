---
license: cc-by-nc-sa-4.0
language:
- en
pipeline_tag: text-to-speech
library_name: drifting-tts
datasets:
- VoiceHub/echo-clones-10kh-en
tags:
- text-to-speech
- autoregressive
- drifting
---

# Drifting TTS English (ARDrift)

An English text-to-speech model trained from scratch on 10,000 hours of synthetic speech
([VoiceHub/echo-clones-10kh-en](https://huggingface.co/datasets/VoiceHub/echo-clones-10kh-en)). It is the English
version of ARDrift, the autoregressive model of [kadirnar/drifting-tts](https://github.com/kadirnar/drifting-tts).

- **How it generates:** a causal transformer produces the mel spectrogram 43 ms at a time. A small head turns noise
  into each piece in one pass. It is trained with the drifting objective
  ([Deng et al., 2026](https://arxiv.org/abs/2602.04770)), with Kyutai's learned temperature.
- **Text:** an English character frontend reads numbers, dates, money, clock times, abbreviations and acronyms itself.
  There is no phonemiser.
- **Voices:** the voices are those of the training data, chosen by speaker ID. There is no voice cloning.

**Demo:** [Vyvo/drifting-tts-en-demo](https://huggingface.co/spaces/Vyvo/drifting-tts-en-demo)

## Files

| file | what | size |
|---|---|---|
| `ardrift_en.pt` | the acoustic model: 36 M text encoder, 121 M backbone, 20 M head | 679 MB |
| `bigvgan_en.pt` | its vocoder: NVIDIA BigVGAN-v2 (24 kHz, 100 bands) fine-tuned on this model's own mels | 450 MB |
| `voices.json` | eight voices picked from the 4,000 training speakers: a label and a speaker ID each | 1 kB |

The acoustic model's duration predictor was re-fitted to the mean duration of every token. With it, the model pauses
at punctuation as long as the recordings do: 1.09 s per sentence against 1.19 s. The regressor it was trained with
paused 0.45 s.

## Usage

```bash
pip install "drifting-tts[bigvgan] @ git+https://github.com/kadirnar/drifting-tts"   # needs a CUDA build of PyTorch
```

```python
import json

import soundfile as sf
from huggingface_hub import hf_hub_download

from drifting_tts.synthesize import Synthesizer

repo = "Vyvo/drifting-tts-en"
tts = Synthesizer(hf_hub_download(repo, "ardrift_en.pt"), "cuda", vocoder=hf_hub_download(repo, "bigvgan_en.pt"))
voices = json.load(open(hf_hub_download(repo, "voices.json")))["voices"]
wav, info = tts("Hello! It's 7:45, and the meeting moved to the 21st.", speaker=voices[0]["id"], temperature=0.5)
sf.write("hello.wav", wav.numpy(), 24000)
```

Use noise temperature 0.5 and no guidance (`cfg_scale=1`, the default).

## Results

The 500 held-out sentences come from a source shard that is not among the 10,000 hours (val 200 + dev 300). The texts
were never trained on; the voices were. Whisper large-v3 transcribes, UTMOSv2 predicts naturalness, and similarity is
WavLM-ECAPA against the recording passed through the same vocoder.

| | WER | CER | UTMOSv2 | similarity |
|---|---|---|---|---|
| the recordings through this vocoder | 1.57% | 0.73% | 3.294 | – |
| **this model** | **1.39%** | **0.61%** | **3.236** | 0.846 |
| the same model with Vocos v2 | 1.42% | 0.64% | 2.930 | – |

The model is as intelligible as its synthetic training data, and its naturalness is close to the recordings' through
the same vocoder. Details: [docs/ENGLISH.md](https://github.com/kadirnar/drifting-tts/blob/main/docs/ENGLISH.md).

## Limits

- **Synthetic data:** the model learns the voices and artefacts of the TTS model that generated its training data.
- **The test set is from the same synthetic corpus:** the texts were never seen, the voices were. There is no
  comparison with other English TTS systems on a common benchmark.
- **Speakers by ID only:** there is no voice cloning from a reference recording.
- **Naturalness was measured only with automatic scores**, not by listeners.

## License

CC BY-NC-SA 4.0: non-commercial use only, and derivatives must share alike. The training speech was generated with
[EchoTTS (jordand/echo-tts-base)](https://huggingface.co/jordand/echo-tts-base), which is licensed CC-BY-NC-SA-4.0.
The vocoder is fine-tuned from [NVIDIA BigVGAN-v2](https://huggingface.co/nvidia/bigvgan_v2_24khz_100band_256x)
(MIT). The code is MIT ([kadirnar/drifting-tts](https://github.com/kadirnar/drifting-tts)).
