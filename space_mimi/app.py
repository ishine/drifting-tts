"""Drifting TTS, English on Mimi latents (a private test Space of its own): ARDrift trained from scratch on 10,000 h of
synthetic English speech in the latent space of Kyutai's Mimi, Pocket TTS's codec (docs/MIMI.md), decoded with the
released Mimi decoder. Runs on its own too (``python app.py``).

Files beside this app (``scripts/deploy_mimi_space.sh``): ``ardrift_en_mimi.pt`` (the model), ``mimi.safetensors``
(Mimi's released weights, CC BY 4.0, read through ``$DRIFTING_TTS_MIMI_WEIGHTS``: no Hub token needed for the gated
repo) and ``english_voices.json`` (the voices offered and the measured results).
"""

import json
import os
import time
from pathlib import Path

import gradio as gr
import numpy as np

from drifting_tts.audio import SAMPLE_RATE
from drifting_tts.latents.mimi import WEIGHTS_ENV
from drifting_tts.synthesize import Synthesizer

try:
    import spaces

    gpu = spaces.GPU(duration=60)
except ImportError:  # running locally
    gpu = lambda f: f  # noqa: E731

HERE = Path(__file__).parent
DEVICE = os.environ.get("DEMO_DEVICE", "cuda")
if (HERE / "mimi.safetensors").is_file():
    os.environ.setdefault(WEIGHTS_ENV, str(HERE / "mimi.safetensors"))
CFG = json.loads((HERE / "english_voices.json").read_text())
VOICES = {v["label"]: v for v in CFG["voices"]}
synth = Synthesizer(HERE / "ardrift_en_mimi.pt", DEVICE)  # a Mimi-latent model: the released Mimi decoder
MAX_CHARS = 600

EXAMPLES = [
    ["Honestly, I didn't expect the weather to turn this quickly, but here we are, stuck inside again."],
    ["Hello! It's 7:45, and the 90s called. They want their dial-up modem back."],
    ["I spent $12.50 on coffee this week, which is honestly about 20% of my budget."],
    ["Dr. Smith said the meeting moved to the 21st, so we'll have to reschedule everything."],
    ["Once upon a time, in a small village by the sea, there lived an old storyteller who knew every tale ever told."],
]


@gpu
def generate(text, voice, temperature, rate, seed):
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
    wav, info = synth(text, speaker=int(v["id"]), cfg_scale=1.0, temperature=float(temperature),
                      length_scale=1.0 / float(rate), seed=int(seed))
    took = time.time() - t0
    stats = f"{info['seconds']:.1f} s ses, {took:.2f} s'de üretildi. Normalleştirilmiş metin: *{normalized}*"
    return (SAMPLE_RATE, np.clip(wav.float().numpy(), -1, 1)), stats


with gr.Blocks(title="Drifting TTS: English, Mimi (test)") as demo:
    gr.Markdown(
        "# ARDrift: İngilizce, Mimi latent'leri (özel test)\n"
        "Kendi otoregresif modelimiz, İngilizce test Space'indeki modelle aynı veri (10.000 saat sentetik İngilizce, "
        "VoiceHub/echo-clones-10kh-en), aynı mimari ve aynı seslerle; tek fark, mel yerine Kyutai'nin Pocket TTS "
        "codec'i Mimi'nin latent'lerini üretmesi. Ses, Kyutai'nin yayımladığı Mimi dekoderiyle çözülür "
        "([kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts), CC BY 4.0). Bir deney: yayın değil."
    )
    with gr.Row():
        with gr.Column(scale=3):
            text = gr.Textbox(label="İngilizce metin", lines=4, max_length=MAX_CHARS, value=EXAMPLES[0][0])
            voice = gr.Dropdown(list(VOICES), value=next(iter(VOICES)), label="Ses")
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
    gr.Markdown(CFG.get("results_md", ""))
    button.click(generate, [text, voice, temperature, rate, seed], [audio, stats])

if __name__ == "__main__":
    demo.queue(max_size=20).launch()
