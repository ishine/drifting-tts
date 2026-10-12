"""Drifting TTS, private test Space: every model under test in one Space, one tab each.

- "Yeni sesler": ``voices_app.py``, ARDrift fine-tuned with six voices learned from 16 kHz recordings, with the mel
  bandwidth extension (drifting_tts/bwe.py).
- "ARDrift ve v3.2": the public demo's app (``space/app.py``, staged as ``demo_app.py``), which offers ARDrift
  (docs/AR_DRIFT.md) next to v3.2 and v3.1 when ``ardrift.pt`` is beside it.
- "Sonuçlar": the measured results of both.

``scripts/deploy_test_space.sh`` stages the two apps beside this one with their files.
"""

import demo_app
import gradio as gr
import voices_app

ARDRIFT_RESULTS_MD = """
### ARDrift ve v3.2 (Freya-TR-Eval, 495 cümle)

Eğitimde görülmemiş 495 günlük cümle; Whisper large-v3 8 kHz, UTMOSv2. ARDrift: Vocos v2, gürültü sıcaklığı 0.5,
guidance yok. v3.2: Vocos v2, sıcaklık 0.3, α 2. WER / UTMOSv2:

| ses | ARDrift | v3.2 (herkese açık demo) |
|---|---|---|
| stüdyo | **0.84%** / 3.006 | 1.33% / 3.021 |
| erkek | **1.79%** / 2.899 | 2.28% / 2.896 |
| kadın | **3.53%** / 2.665 | 3.99% / 2.722 |

- ARDrift her seste daha az kelime hatası yapıyor (stüdyoda anlamlı fark); ses kalitesi stüdyo ve erkek seste aynı,
  kadın seste 0.06 düşük.
- Tonlama: stüdyo sesinin 100 kaydında perde eğrisine benzerlik (DTW F0 r) ARDrift 0.633, v3.2 0.579; bu projede
  ölçülen en yakın sonuç. Perde aralığı ise daha dar (kadın seste 2.32'ye karşı 3.04 yarım ton).
- İlk ses: ARDrift 36 ms, v3.2 6-8 ms (RTX 5090).

### Yeni sesler
"""

with gr.Blocks(title="Drifting TTS: özel test") as demo:
    gr.Markdown(
        "# Drifting TTS: özel test\n"
        "Test edilen bütün modeller tek yerde. **Yeni sesler:** ARDrift'in altı yeni sesle eğitilmiş hali. "
        "**ARDrift ve v3.2:** kendi otoregresif modelimiz ARDrift, herkese açık demodaki v3.2 ve v3.1 ile yan yana. "
        "**Sonuçlar:** ölçümler."
    )
    with gr.Tab("Yeni sesler"):
        voices_app.demo.render()
    with gr.Tab("ARDrift ve v3.2"):
        demo_app.demo.render()
    with gr.Tab("Sonuçlar"):
        gr.Markdown(ARDRIFT_RESULTS_MD + voices_app.results_md())

if __name__ == "__main__":
    demo.queue(max_size=20).launch()
