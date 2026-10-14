"""Kyutai's Mimi as used by Pocket TTS (CC BY 4.0): 32-d continuous latents at 12.5 Hz, 24 kHz in and out, causal.

The codec of Kyutai's Pocket TTS, whose autoregressive model with a one-step drifting head works in this latent space
(https://kyutai.org/blog/2026-09-28-pocket-tts-drifting/). A SEANet encoder (strides 6·5·4: 200 Hz) with a 2-layer
transformer, a learned 16x downsampling to the 32-d latent (the VAE's mean; no quantization), and the mirror-image
decoder. The model code is the ``pocket-tts`` package (Kyutai, MIT; ``pip install "drifting-tts[mimi]"``); the weights
are the Mimi part of ``kyutai/pocket-tts`` ``languages/english_2026-09/model.safetensors`` (gated: accept its terms on
the Hub once), pinned. On 100 held-out English recordings a round trip scores UTMOSv2 3.132 (recordings 3.259, the
recording's mel through Vocos v2 2.926; docs/MIMI.md).
"""

from __future__ import annotations

import torch
from torch import Tensor

from .base import AudioBackend

REPO, FILE, REVISION = "kyutai/pocket-tts", "languages/english_2026-09/model.safetensors", \
    "983151f13aaeab1b13c1e5e3c2c383d49a9edf3f"
CONFIG = "english_2026-09.yaml"  # pocket-tts's config of those weights (its `mimi` section)


def load_mimi(device: str = "cpu"):
    """Pocket TTS's Mimi with its released weights, in eval mode, ready for whole-utterance decoding."""
    import safetensors.torch
    from huggingface_hub import hf_hub_download
    from pocket_tts.models.mimi import build_mimi
    from pocket_tts.modules.stateful_module import StatefulModule
    from pocket_tts.utils.config import CONFIGS_DIR, load_config

    mimi = build_mimi(load_config(CONFIGS_DIR / CONFIG).mimi)
    bundle = safetensors.torch.load_file(hf_hub_download(REPO, FILE, revision=REVISION))
    mimi.load_state_dict({k.removeprefix("mimi."): v for k, v in bundle.items() if k.startswith("mimi.")}, strict=True)
    for name, module in mimi.named_modules():  # stateful layers find their state by name (as stamp_state_names)
        if isinstance(module, StatefulModule):
            module._module_absolute_name = name
    return mimi.to(device).eval()


class MimiBackend(AudioBackend):
    name = "mimi"
    input_rate = output_rate = 24000
    frame_rate = 12.5
    dim = 32
    causal = True

    def __init__(self, device: str = "cuda", decoder: dict | None = None):
        """``decoder``: a fine-tuned decoder (``decoder_ft.pt`` of ``finetune-vocoder``, loaded with
        :func:`drifting_tts.latents.vocoder.load_decoder_checkpoint`) in place of the released one."""
        self.device = device
        self.model = load_mimi(device)
        if decoder is not None:
            self.trainable_decoder().load_state_dict(decoder["decoder"])

    @torch.no_grad()
    def encode(self, wav: Tensor, sr: int) -> Tensor:
        x = self._input(wav, sr)  # [B, 1, S]
        z = self.model.encode_to_latent(x)  # [B, T, 32], T = ceil(S / 1920)
        return z.transpose(1, 2).float()

    @torch.no_grad()
    def decode(self, latent: Tensor) -> Tensor:
        from pocket_tts.modules.stateful_module import init_states

        latent = latent.float().to(self.device)
        b, _, t = latent.shape
        state = init_states(self.model, batch_size=b, sequence_length=max(10000, 16 * t + 16))
        out = self.model.decode_from_latent(latent.transpose(1, 2), state)[:, 0]
        return out[:, : t * self.hop_out].clamp(-1, 1)

    def trainable_decoder(self) -> torch.nn.Module:
        """Everything between the latent and the waveform: the latent's projection, the 2x transposed-convolution
        upsampler, the 2-layer transformer and the SEANet decoder (``finetune-vocoder``, ``decoder_ft.pt``)."""
        m = self.model
        return torch.nn.ModuleDict({"quantizer": m.quantizer, "upsample": m.upsample,
                                    "decoder_transformer": m.decoder_transformer, "decoder": m.decoder})

    def decode_train(self, latent: Tensor) -> Tensor:
        """:meth:`decode` with gradients, unclamped: the convolutions start from fresh (zero) streaming states as in
        :meth:`decode`; the transformer runs its stateless causal path (the same attention window as the KV cache)."""
        from pocket_tts.modules.conv import StreamingConv1d, StreamingConvTranspose1d

        m, (b, _, t) = self.model, latent.shape
        state = {name: mod.init_state(b, 0) for name, mod in m.named_modules()
                 if isinstance(mod, (StreamingConv1d, StreamingConvTranspose1d))}
        x = m._to_encoder_framerate(m.quantizer(latent), state)
        (x,) = m.decoder_transformer(x, None)
        return m.decoder(x, state)[:, 0, : t * self.hop_out]
