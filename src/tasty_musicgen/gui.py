import streamlit as st
import torchaudio
import torch
import matplotlib.pyplot as plt
from tasty_musicgen.model import get_model, make_inference


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


st.write(
    """
# MusicGEN GUI

This is a simple GUI for the MusicGEN model.      
"""
)

device = st.sidebar.selectbox(
    "Select the device to use",
    ("cpu", "cuda"),
    key="device",
)

model_name = st.sidebar.selectbox(
    "Select the model to use",
    ("csc-unipd/tasty-musicgen-small", "facebook/musicgen-small"),
)

prompt = st.text_input(
    "Enter a prompt for the music generation",
    "A happy tune",
)

duration = st.slider(
    "Enter the duration of the music in seconds",
    min_value=1.0,
    max_value=90.0,
    value=10.0,
    step=0.5,
)

button = st.button("Generate Music")

if button:

    with st.spinner("Generating music..."):
        model = get_model(device=device, model=model_name)
        # Generate audio
        music = make_inference(model, prompt, duration=duration)

    y = music.squeeze(0).cpu()
    # Plot the mel spectrogram
    fig = plot_spectrogram(y, model.sample_rate)
    st.pyplot(fig)

    st.audio(
        y.numpy(),
        format="audio/wav",
        sample_rate=model.sample_rate,
    )
