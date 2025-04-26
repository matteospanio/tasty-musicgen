import streamlit as st
from tasty_musicgen.gui.utils import load_model, draw_spectrogram, draw_sidebar
from tasty_musicgen.model import make_audio_from_text

draw_sidebar()

st.header("Text conditioning")

# st.badge(
#     "HF Model",
#     icon="🤗",
#     # href="https://huggingface.co/docs/transformers/main/model_doc/musicgen#text-conditional-generation",
# )

st.markdown(
    """
[:violet-badge[:hugging_face: HF guide]](https://huggingface.co/docs/transformers/main/model_doc/musicgen#text-conditional-generation)

Abbiamo visto come creare musica casuale, ma vorremmo fare di più,
idealmente è più interessante cercare una maniera di generare musica avendo la possibilità di
spiegare al computer cosa vorremmo ottenere.

Questo fenomeno è chiamato [**Text conditioning**](https://serp.ai/posts/conditional-text-generation/),
ovvero la possibilità di generare musica a partire da un testo che descrive il brano che vogliamo ottenere.
In questo caso, il modello di intelligenza artificiale è in grado di generare un brano musicale
a partire da un testo che descrive il brano stesso.

"""
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
    device = st.session_state.device
    model_name = st.session_state.model_name
    model = load_model(device=device, model=model_name)
    with st.spinner("Generating music..."):
        # Generate audio
        music = make_audio_from_text(model, prompt, duration=duration)

    y = music.squeeze(0).cpu()
    draw_spectrogram(y, model)

    st.audio(
        y.numpy(),
        format="audio/wav",
        sample_rate=model.sample_rate,
    )
