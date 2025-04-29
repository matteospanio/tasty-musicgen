import streamlit as st
from tasty_musicgen.model import (
    make_audio_from_given_melody_and_text,
)
from tasty_musicgen.gui.utils import load_model, draw_spectrogram, draw_sidebar
import torchaudio

draw_sidebar(disable_model_selection=True)

st.header("Modificare un audio")
st.markdown(
    """
Il processo creativo di un brano è complesso e richiede molto lavoro, a volte capita di avere delle idee melodiche,
ma non sapere bene come arrangiarle o orchestrarle, oppure di avere un'idea di un brano ma non sapere come
realizzarlo.
L'Intelligenza Artificiale può prendere in input un file audio e rielaborarlo 
"""
)

with st.container(border=True):
    tab_fi, tab_record = st.tabs(["File", "Registrazione"])

    with tab_fi:
        st.write(
            """
            Carica un file audio e scegli una durata per il brano.
            """
        )
        input_audio = st.file_uploader(
            "Upload an audio file",
            type=["wav", "mp3", "ogg"],
            label_visibility="collapsed",
        )

    with tab_record:
        st.info(
            """Il contenuto registrato non verrà salvato, inoltre viene utilizzato solo se
            non è stato caricato alcun file audio.
            Se vuoi usare il file registrato, ma hai già caricato un file audio, cancella gli
            upload.
            """
        )
        st.write(
            """
            Registra un file audio e scegli una durata per il brano.
            """
        )
        input_record = st.audio_input(
            "Record an audio file",
            label_visibility="collapsed",
        )

    col1, col2 = st.columns(2)

    with col1:
        prompt = st.text_area(
            "Enter a prompt for the music generation",
            "A happy tune",
        )

    with col2:
        duration = st.slider(
            "Enter the duration of the music in seconds",
            min_value=1.0,
            max_value=90.0,
            value=10.0,
            step=0.5,
        )
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
        )

    if st.button("Genera musica"):
        device = st.session_state.device
        model_name = st.session_state.model_name
        model = load_model(device=device, model="facebook/musicgen-melody")
        signal, sr = torchaudio.load(input_audio if input_audio else input_record)
        with st.spinner("Generating music..."):
            # Generate audio
            music = make_audio_from_given_melody_and_text(
                model,
                prompt=prompt,
                audio=signal,
                audio_sample_rate=sr,
                duration=duration,
                temperature=temperature,
            )

        y = music.squeeze(0).cpu()
        draw_spectrogram(y, model)

        st.audio(
            y.numpy(),
            format="audio/wav",
            sample_rate=model.sample_rate,
        )
