"""Drifting TTS English demo (public): ARDrift trained from scratch on 10,000 h of synthetic English speech, with
BigVGAN-v2 fine-tuned on its mels and its duration predictor re-fitted to the mean (docs/ENGLISH.md). The weights are
downloaded from the model repo Vyvo/drifting-tts-en; ``scripts/deploy_english_demo.sh`` copies this app and the
package into the Space.
"""

import json
import os
import time

import gradio as gr
import numpy as np
from huggingface_hub import hf_hub_download

from drifting_tts.audio import SAMPLE_RATE
from drifting_tts.synthesize import Synthesizer

try:
    import spaces

    gpu = spaces.GPU(duration=60)
except ImportError:  # running locally
    gpu = lambda f: f  # noqa: E731

REPO = os.environ.get("DRIFTING_TTS_EN_REPO", "Vyvo/drifting-tts-en")
DEVICE = os.environ.get("DEMO_DEVICE", "cuda")
synth = Synthesizer(hf_hub_download(REPO, "ardrift_en.pt"), DEVICE, vocoder=hf_hub_download(REPO, "bigvgan_en.pt"))
VOICES = {v["label"]: v for v in json.loads(open(hf_hub_download(REPO, "voices.json")).read())["voices"]}
MAX_CHARS = 600

EXAMPLES = [
    ["Honestly, I didn't expect the weather to turn this quickly, but here we are, stuck inside again."],
    ["Hello! It's 7:45, and the 90s called. They want their dial-up modem back."],
    ["I spent $12.50 on coffee this week, which is honestly about 20% of my budget."],
    ["Dr. Smith said the meeting moved to the 21st, so we'll have to reschedule everything."],
    ["Once upon a time, in a small village by the sea, there lived an old storyteller who knew every tale ever told."],
]

RESULTS_MD = f"""
### Results

500 held-out sentences, never trained on (Whisper large-v3, UTMOSv2):

| | WER | CER | UTMOSv2 |
|---|---|---|---|
| the recordings through the same vocoder | 1.57% | 0.73% | 3.294 |
| **this model** | **1.39%** | **0.61%** | **3.236** |

- **Synthetic training data.** The model learns the voices and artefacts of the TTS model that generated its
  data. The test texts are new; the voices were seen in training.
- **Voices.** Eight voices from the 4,000 training speakers are offered, chosen by speaker ID. There is no voice
  cloning.
- **Pauses.** The duration predictor is re-fitted to the mean duration of every token, so the model pauses at
  punctuation as long as the recordings do.

**License.** CC BY-NC-SA 4.0, for non-commercial use: the training speech was generated with a CC-BY-NC-SA-4.0 model.
Model: [{REPO}](https://huggingface.co/{REPO}). Code: [kadirnar/drifting-tts](https://github.com/kadirnar/drifting-tts).
"""


@gpu
def generate(text, voice, temperature, rate, seed):
    text = " ".join((text or "").split())
    if not text:
        raise gr.Error("Please enter some English text.")
    if len(text) > MAX_CHARS:
        raise gr.Error(f"Please keep the text under {MAX_CHARS} characters.")
    normalized = synth.text.normalize(text)
    if not normalized:
        raise gr.Error("Nothing to read after normalisation.")
    t0 = time.time()
    wav, info = synth(text, speaker=int(VOICES[voice]["id"]), cfg_scale=1.0, temperature=float(temperature),
                      length_scale=1.0 / float(rate), seed=int(seed))
    took = time.time() - t0
    stats = f"{info['seconds']:.1f} s of audio in {took:.2f} s. Normalised text: *{normalized}*"
    return (SAMPLE_RATE, np.clip(wav.float().numpy(), -1, 1)), stats


with gr.Blocks(title="Drifting TTS: English") as demo:
    gr.Markdown(
        "# Drifting TTS: English\n"
        "An autoregressive text-to-speech model trained from scratch on 10,000 hours of synthetic English speech. A "
        "causal transformer writes the mel spectrogram 43 ms at a time, and a small head draws each piece in one pass "
        "(the drifting objective, [Deng et al., 2026](https://arxiv.org/abs/2602.04770)). The character frontend reads "
        "numbers, money, clock times, abbreviations and acronyms by itself."
    )
    with gr.Row():
        with gr.Column(scale=3):
            text = gr.Textbox(label="English text", lines=4, max_length=MAX_CHARS, value=EXAMPLES[0][0])
            voice = gr.Dropdown(list(VOICES), value=next(iter(VOICES)), label="Voice")
            with gr.Accordion("Settings", open=False):
                temperature = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Noise temperature",
                                        info="0.5 is the measured setting")
                rate = gr.Slider(0.7, 1.4, value=1.0, step=0.05, label="Speaking rate")
                seed = gr.Number(value=0, precision=0, label="Seed")
            button = gr.Button("Speak", variant="primary")
        with gr.Column(scale=2):
            audio = gr.Audio(label="Speech (24 kHz)", type="numpy", autoplay=True)
            stats = gr.Markdown()
    gr.Examples(EXAMPLES, inputs=[text])
    gr.Markdown(RESULTS_MD)
    button.click(generate, [text, voice, temperature, rate, seed], [audio, stats])

if __name__ == "__main__":
    demo.queue(max_size=20).launch()
