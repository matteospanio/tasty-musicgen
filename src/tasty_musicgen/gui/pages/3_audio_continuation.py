import streamlit as st
from tasty_musicgen.model import continue_audio
from tasty_musicgen.gui.utils import load_model, draw_spectrogram, draw_sidebar
import torchaudio

draw_sidebar(disable_model_selection=True)

st.header("Continuazione di un audio")
st.markdown(
"""
"""
)

with st.container(border=True):
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
        signal, sr = torchaudio.load(input_audio)
        with st.spinner("Generating music..."):
            # Generate audio
            music = continue_audio(
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

