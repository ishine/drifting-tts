"""Drifting TTS, English (the "English" tab of the private test Space, app.py): ARDrift trained from scratch on 10,000 h
of synthetic English speech (docs/ENGLISH.md). Runs on its own too (``python english_app.py``).

Files beside this app (``scripts/deploy_test_space.sh``): ``ardrift_en.pt`` (the model), ``english_voices.json``
(the voices offered: label and speaker ID, how they were picked, and the model's measured results) and, optionally,
``bigvgan_en.pt``: BigVGAN-v2 fine-tuned on this model's own mels (issue #70), offered next to Vocos v2.
"""

import json
import os
import time
from pathlib import Path

import gradio as gr
import numpy as np

from drifting_tts.audio import SAMPLE_RATE
from drifting_tts.synthesize import Synthesizer

try:
    import spaces

    gpu = spaces.GPU(duration=60)
except ImportError:  # running locally
    gpu = lambda f: f  # noqa: E731

HERE = Path(__file__).parent
DEVICE = os.environ.get("DEMO_DEVICE", "cuda")
CFG = json.loads((HERE / "english_voices.json").read_text())
VOICES = {v["label"]: v for v in CFG["voices"]}
synth = Synthesizer(HERE / "ardrift_en.pt", DEVICE, vocoder="vocos-v2")
VOCODERS = {"Vocos v2": synth}
if (HERE / "bigvgan_en.pt").is_file():  # first: the default
    VOCODERS = {"BigVGAN-v2, İngilizce ince ayar": synth.variant(vocoder=str(HERE / "bigvgan_en.pt")), **VOCODERS}
MAX_CHARS = 600

EXAMPLES = [
    ["Honestly, I didn't expect the weather to turn this quickly, but here we are, stuck inside again."],
    ["Hello! It's 7:45, and the 90s called. They want their dial-up modem back."],
    ["I spent $12.50 on coffee this week, which is honestly about 20% of my budget."],
    ["Dr. Smith said the meeting moved to the 21st, so we'll have to reschedule everything."],
    ["Once upon a time, in a small village by the sea, there lived an old storyteller who knew every tale ever told."],
]


def results_md() -> str:
    return CFG.get("results_md", "")


@gpu
def generate(text, voice, vocoder, temperature, rate, seed):
    text = " ".join((text or "").split())
    if not text:
        raise gr.Error("Lütfen İngilizce bir metin girin.")
    if len(text) > MAX_CHARS:
        raise gr.Error(f"Metin en fazla {MAX_CHARS} karakter olabilir.")
    normalized = synth.text.normalize(text)
    if not normalized:
        raise gr.Error("Normalleştirmeden sonra okunacak bir şey kalmadı.")
    v = VOICES[voice]
    t0 = time.time()
    wav, info = VOCODERS[vocoder](text, speaker=int(v["id"]), cfg_scale=1.0, temperature=float(temperature),
                      length_scale=1.0 / float(rate), seed=int(seed))
    took = time.time() - t0
    stats = f"{info['seconds']:.1f} s ses, {took:.2f} s'de üretildi. Normalleştirilmiş metin: *{normalized}*"
    return (SAMPLE_RATE, np.clip(wav.float().numpy(), -1, 1)), stats


with gr.Blocks(title="Drifting TTS: English (test)") as demo:
    gr.Markdown(
        "# ARDrift: İngilizce (özel test)\n"
        "Kendi otoregresif modelimiz, 10.000 saatlik sentetik İngilizce konuşmayla sıfırdan eğitildi "
        "(VoiceHub/echo-clones-10kh-en, 4.000 konuşmacı). Harf tabanlı İngilizce metin katmanı sayıları, saatleri, "
        "parayı ve kısaltmaları kendisi okur."
    )
    with gr.Row():
        with gr.Column(scale=3):
            text = gr.Textbox(label="İngilizce metin", lines=4, max_length=MAX_CHARS, value=EXAMPLES[0][0])
            voice = gr.Dropdown(list(VOICES), value=next(iter(VOICES)), label="Ses")
            vocoder = gr.Radio(list(VOCODERS), value=next(iter(VOCODERS)), label="Vocoder",
                               visible=len(VOCODERS) > 1)
            with gr.Accordion("Ayarlar", open=False):
                temperature = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Gürültü sıcaklığı",
                                        info="0.5 ölçülen ayar")
                rate = gr.Slider(0.7, 1.4, value=1.0, step=0.05, label="Konuşma hızı")
                seed = gr.Number(value=0, precision=0, label="Seed")
            button = gr.Button("Seslendir", variant="primary")
        with gr.Column(scale=2):
            audio = gr.Audio(label="Konuşma (24 kHz)", type="numpy", autoplay=True)
            stats = gr.Markdown()
    gr.Examples(EXAMPLES, inputs=[text])
    gr.Markdown(results_md())
    button.click(generate, [text, voice, vocoder, temperature, rate, seed], [audio, stats])

if __name__ == "__main__":
    demo.queue(max_size=20).launch()
