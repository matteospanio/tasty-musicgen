import streamlit as st
from tasty_musicgen.model import get_model
from tasty_musicgen.utils import plot_spectrogram


@st.cache_resource
def load_model(device: str, model: str):
    """
    Get the model from the model name and device.
    """
    return get_model(device=device, model=model)


def draw_spectrogram(y, model):
    # Plot the mel spectrogram
    fig = plot_spectrogram(y, model.sample_rate)
    st.pyplot(fig)


def draw_sidebar(disable_model_selection: bool = False):
    with st.sidebar:
        st.image("assets/logo_light.png")
        st.divider()

    st.sidebar.selectbox(
        "Select the device to use",
        ("cpu", "cuda"),
        key="device",
    )

    st.sidebar.selectbox(
        "Select the model to use",
        (
            "csc-unipd/tasty-musicgen-small",
            "facebook/musicgen-small",
            "facebook/musicgen-melody",
            "facebook/musicgen-stereo-small",
        ),
        key="model_name",
        disabled=disable_model_selection,
        index=2 if disable_model_selection else 0,
    )
