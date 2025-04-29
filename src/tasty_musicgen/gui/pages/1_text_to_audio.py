import streamlit as st
from tasty_musicgen.gui.utils import load_model, draw_spectrogram, draw_sidebar, draw_text_blackbox
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

with st.container(border=True):
    st.title("Audio da testo")
    draw_text_blackbox()
