from typing import Tuple
from audiocraft.data.audio import audio_write
from audiocraft.models import MusicGen
from torch import Tensor

AudioArray = Tensor | Tuple[Tensor, Tensor]

def get_model() -> MusicGen:
    musicgen = MusicGen.get_pretrained("csc-unipd/tasty-musicgen-small")
    return musicgen

def make_inference(synthesiser:MusicGen, prompt: str, duration: float = 30.0) -> AudioArray:
    """
    Generate audio from a text prompt using the synthesiser.
    """
    synthesiser.set_generation_params(duration=duration)

    # Generate audio
    music = synthesiser.generate([prompt], progress=True)
    return music

def save_audio(music: AudioArray, out_file: str, model: MusicGen):
    audio_write(
        out_file,
        music.cpu(),
        model.sample_rate,
        strategy="loudness",
        loudness_compressor=True,
    )

def main() -> None:
    # Load the model
    synthesiser = get_model()

    # Define the prompt
    prompt = "A happy birthday song for a child played on a trumpet."

    # Generate audio
    music = make_inference(synthesiser, prompt)

    # Save the audio to a file
    save_audio(music, "happy_birthday.wav", synthesiser)
