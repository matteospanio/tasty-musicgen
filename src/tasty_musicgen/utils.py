from torch import Tensor
from audiocraft.data.audio import audio_write
from audiocraft.models import MusicGen


def save_audio(music: Tensor, out_file: str, model: MusicGen) -> None:
    audio_write(
        out_file,
        music.cpu(),
        model.sample_rate,
        strategy="loudness",
        loudness_compressor=True,
    )
