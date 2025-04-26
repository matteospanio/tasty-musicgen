import streamlit as st
from tasty_musicgen.model import make_random_audio
from tasty_musicgen.gui.utils import load_model, draw_spectrogram, draw_sidebar

st.set_page_config(
    page_title="MusicGEN",
    page_icon="🎵",
)

st.write(
    """
# Musica e Intelligenza Artificiale

This is a simple GUI for the MusicGEN model.      
"""
)

draw_sidebar()

duration = st.slider(
    "Imposta la durata della musica in secondi",
    min_value=1.0,
    max_value=90.0,
    value=10.0,
    step=0.5,
)

btn = st.button("Genera musica casuale")
if btn:
    model_name = st.session_state.model_name
    device = st.session_state.device
    model = load_model(device=device, model=model_name)
    with st.spinner("Generazione in corso..."):
        # Generate audio
        music = make_random_audio(model, duration=duration, temperature=1.0)

    y = music.squeeze(0).cpu()
    draw_spectrogram(y, model)
    st.audio(
        y.numpy(),
        format="audio/wav",
        sample_rate=model.sample_rate,
    )
