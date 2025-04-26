import streamlit as st
from tasty_musicgen.model import get_model, make_inference
from tasty_musicgen.utils import plot_spectrogram


@st.cache_resource
def load_model(device: str, model: str):
    """
    Get the model from the model name and device.
    """
    return get_model(device=device, model=model)


st.write(
    """
# MusicGEN GUI

This is a simple GUI for the MusicGEN model.      
"""
)

with st.sidebar:
    st.image("assets/Csc_oriz_bianco-2.png")
    st.divider()

device = st.sidebar.selectbox(
    "Select the device to use",
    ("cpu", "cuda"),
    key="device",
)

model_name = st.sidebar.selectbox(
    "Select the model to use",
    (
        "csc-unipd/tasty-musicgen-small",
        "facebook/musicgen-small",
        "facebook/musicgen-medium",
        "facebook/musicgen-melody",
        "facebook/musicgen-stereo-small",
    ),
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
    model = load_model(device=device, model=model_name)
    with st.spinner("Generating music..."):
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
