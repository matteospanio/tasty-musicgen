from rich.prompt import Prompt
from rich.console import Console
import uuid
import click
from pathlib import Path
from rich.progress import track

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


@click.command()
@click.argument("prompts", nargs=-1)
@click.option(
    "--length",
    "-l",
    default=30.0,
    help="Length of the generated audio in seconds",
    type=float,
)
@click.option(
    "--device",
    "-d",
    default="cpu",
    help="Device to run the model on (e.g., 'cpu', 'cuda')",
    type=click.Choice(["cpu", "cuda"]),
)
@click.option(
    "--output-folder",
    "-o",
    default="output",
    help="Folder to save the generated audio files",
    type=click.Path(
        exists=True,
        file_okay=False,
        writable=True,
        path_type=Path,
    ),
)
def batch(prompts: list[str], length: float, device: str, output_folder: Path) -> None:
    console = Console()

    with console.status("[bold green]Loading model...[/]", spinner="clock") as status:
        # Load the model
        synthesiser = get_model(device=device)
        status.stop()

    # draw progress bar
    for prompt in track(prompts, description="Generating music..."):
        destination = output_folder / f"{uuid.uuid1()}"
        destination.mkdir(parents=True, exist_ok=True)

        with open(destination / "prompt.txt", "w") as f:
            f.write(prompt)

        for i in range(3):
            # Generate audio
            music = make_audio_from_text(synthesiser, prompt, float(length))

            # Save the audio to a file
            save_audio(music.squeeze(0), f"{str(destination)}/audio_{i}", synthesiser)
