from rich.prompt import Prompt
from rich.console import Console
import uuid

from tasty_musicgen.model import get_model, make_audio_from_text
from tasty_musicgen.utils import save_audio


def main() -> None:
    console = Console()

    with console.status("[bold green]Loading model...[/]", spinner="clock") as status:
        # Load the model
        synthesiser = get_model(device="cpu")
        status.stop()

    # Define the prompt
    prompt = Prompt.ask("Enter a prompt for the music generation", console=console)
    duration = Prompt.ask(
        "Enter the duration of the music in seconds",
        default=30.0,
        console=console,
        show_default=True,
    )

    # Generate audio
    music = make_audio_from_text(synthesiser, prompt, float(duration))

    # Save the audio to a file
    save_audio(music.squeeze(0), f"{uuid.uuid1()}", synthesiser)
