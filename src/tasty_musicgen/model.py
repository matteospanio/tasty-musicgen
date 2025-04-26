from typing import Literal
from audiocraft.models import MusicGen
from torch import Tensor


def get_model(
    model: str = "csc-unipd/tasty-musicgen-small",
    device: Literal["cpu", "cuda"] = "cuda",
) -> MusicGen:
    musicgen = MusicGen.get_pretrained(model, device=device)
    return musicgen


def make_random_audio(
    synthesiser: MusicGen,
    duration: float = 30.0,
    temperature: float = 1.0,
) -> Tensor:
    """
    Generate random audio using the synthesiser.
    """
    synthesiser.set_generation_params(
        duration=duration,
        extend_stride=0.5,
        temperature=temperature,
    )

    # Generate audio
    music = synthesiser.generate_unconditional(1, progress=True)
    return music


def make_audio_from_text(
    synthesiser: MusicGen,
    prompt: str,
    duration: float = 30.0,
    temperature: float = 1.0,
) -> Tensor:
    """
    Generate audio from a text prompt using the synthesiser.
    """
    synthesiser.set_generation_params(
        duration=duration,
        extend_stride=0.5,
        temperature=temperature,
    )

    # Generate audio
    music = synthesiser.generate([prompt], progress=True)
    return music


def make_audio_from_given_melody_and_text(
    synthesiser: MusicGen,
    prompt: str,
    audio: Tensor,
    audio_sample_rate: int,
    duration: float = 30.0,
    temperature: float = 1.0,
) -> Tensor:
    """
    Generate audio from a text prompt using the synthesiser.
    """
    synthesiser.set_generation_params(
        duration=duration,
        extend_stride=0.5,
        temperature=temperature,
    )

    # Generate audio
    music = synthesiser.generate_with_chroma(
        melody_wavs=audio,
        melody_sample_rate=audio_sample_rate,
        descriptions=[prompt],
        progress=True,
    )
    return music


def continue_audio(
    synthesiser: MusicGen,
    prompt: str | None,
    audio: Tensor,
    audio_sample_rate: int,
    duration: float = 30.0,
    temperature: float = 1.0,
) -> Tensor:
    """
    Continue audio from a text prompt using the synthesiser.
    """
    synthesiser.set_generation_params(
        duration=duration,
        extend_stride=0.5,
        temperature=temperature,
    )

    # Generate audio
    music = synthesiser.generate_continuation(
        prompt=audio,
        prompt_sample_rate=audio_sample_rate,
        descriptions=[prompt] if prompt else None,
        progress=True,
    )
    return music
