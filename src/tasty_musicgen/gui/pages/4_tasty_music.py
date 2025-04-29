import streamlit as st
from tasty_musicgen.gui.utils import draw_sidebar, load_model, draw_spectrogram
from tasty_musicgen.model import make_audio_from_text

draw_sidebar(disable_model_selection=True)

st.header("Ma che sapore ha?")

with st.container(border=True):
    st.title("Audio saporito")
    col1, col2 = st.columns(2)
    with col1:
        prompt = st.text_area(
            "Scrivi un testo per la generazione della musica",
            "A salty tune",
        )
        taste = st.selectbox(
            "Scegli il sapore della musica",
            ("Salty", "Sweet", "Sour", "Bitter"),
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
        device = st.session_state.device
        model = load_model(device=device, model="csc-unipd/tasty-musicgen-small")
        with st.spinner("Generazione in corso..."):
            # Generate audio
            music = make_audio_from_text(model, f"{prompt}. {taste} taste.", duration=duration, temperature=temperature)

        y = music.squeeze(0).cpu()
        draw_spectrogram(y, model)
        st.audio(
            y.numpy(),
            format="audio/wav",
            sample_rate=model.sample_rate,
        )
