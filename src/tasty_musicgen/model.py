from typing import Literal
from audiocraft.models import MusicGen
from torch import Tensor


def get_model(
    model: str = "csc-unipd/tasty-musicgen-small",
    device: Literal["cpu", "cuda"] = "cuda",
) -> MusicGen:
    musicgen = MusicGen.get_pretrained(model, device=device)
    return musicgen


def make_inference(
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
