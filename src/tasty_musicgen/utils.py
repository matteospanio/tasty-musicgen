from torch import Tensor
from audiocraft.data.audio import audio_write
from audiocraft.models import MusicGen
import torchaudio
import matplotlib.pyplot as plt
import torch


def save_audio(music: Tensor, out_file: str, model: MusicGen) -> None:
    audio_write(
        out_file,
        music.cpu(),
        model.sample_rate,
        strategy="loudness",
        loudness_compressor=True,
    )


# Function to generate and plot mel spectrogram
def plot_spectrogram(waveform, sample_rate, n_fft=400, hop_length=160):
    # Generate Mel Spectrogram
    spectrogram = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
    )(waveform)

    # Convert to decibels
    spectrogram_db = torchaudio.transforms.AmplitudeToDB()(spectrogram)

    freq_bins = spectrogram_db.shape[-2]
    freqs = torch.linspace(0, sample_rate // 2, freq_bins)

    # Plot
    fig = plt.figure(figsize=(10, 4))
    plt.imshow(
        spectrogram_db.squeeze().numpy(),
        aspect="auto",
        origin="lower",
        extent=[0, waveform.shape[-1] / sample_rate, freqs[0], freqs[-1]],
        cmap="viridis",
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.tight_layout()
    return fig
