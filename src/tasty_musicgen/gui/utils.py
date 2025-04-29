import streamlit as st
from tasty_musicgen.model import get_model, make_random_audio, make_audio_from_text
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
        ("cuda", "cpu"),
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

def gen_button(btn: bool, temperature: float = 1.0, duration: float = 5.0):
    if btn:
        model_name = st.session_state.model_name
        device = st.session_state.device
        model = load_model(device=device, model=model_name)
        with st.spinner("Generazione in corso..."):
            # Generate audio
            music = make_random_audio(model, duration=duration, temperature=temperature)

        y = music.squeeze(0).cpu()
        st.audio(
            y.numpy(),
            format="audio/wav",
            sample_rate=model.sample_rate,
        )

def shallow_blackbox():
    st.markdown("<h4 style='text-align: center;'>Genera musica casuale</h4>", unsafe_allow_html=True)

    with st.columns(3)[1]:
        st.image("https://media.tenor.com/i_L5KauoCcoAAAAj/dice.gif")
        btn = st.button("Tira i dadi")

    gen_button(btn)



def draw_blackbox():
    st.markdown("<h4 style='text-align: center;'>Genera musica casuale</h4>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        temperature = st.slider(
            "Imposta la temperatura",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.05,
        )

    with col2:
        duration = st.slider(
            "Imposta la durata della musica in secondi",
            min_value=1.0,
            max_value=90.0,
            value=10.0,
            step=0.5,
        )

    btn = st.button("Genera musica casuale")
    gen_button(btn, temperature=temperature, duration=duration)

def draw_text_blackbox():
    col1, col2 = st.columns(2)
    with col1:
        prompt = st.text_area(
            "Scrivi un testo per la generazione della musica",
            "A happy tune",
        )

    with col2:
        duration = st.slider(
            "Imposta la durata della musica in secondi",
            min_value=1.0,
            max_value=90.0,
            value=10.0,
            step=0.5,
        )

        temperature = st.slider(
            "Imposta la temperatura",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.05,
        )

    btn = st.button("Genera musica")
    if btn:
        model_name = st.session_state.model_name
        device = st.session_state.device
        model = load_model(device=device, model=model_name)
        with st.spinner("Generazione in corso..."):
            # Generate audio
            music = make_audio_from_text(model, prompt, duration=duration, temperature=temperature)

        y = music.squeeze(0).cpu()
        draw_spectrogram(y, model)
        st.audio(
            y.numpy(),
            format="audio/wav",
            sample_rate=model.sample_rate,
        )