import streamlit as st
from tasty_musicgen.gui.utils import draw_sidebar, shallow_blackbox, draw_blackbox

st.set_page_config(
    page_title="MusicGEN",
    page_icon="🎵",
)

draw_sidebar()

st.write(
"""
# Musica e Intelligenza Artificiale
"""
)

st.image(
    "assets/word_cloud.png",
    caption="Nuvola di parole frequenti usate nei discorsi sull'AI."
)


with st.container(border=True):
    shallow_blackbox()

st.markdown(
"""
Questa AI è detta blackbox perché non sappiamo esattamente come funzioni e non possiamo controllarla.

##### Cosa succede quando tiro i dadi? 🎲

L'AI genera casualmente un evento musicale.
"""
)

st.divider()

st.markdown(
"""
Mi piacerebbe controllare i dadi! 🎲

- Banalmente, mi piacerebbe controllare quanto tempo dura la musica.
- Vorrei anche controllare altre qualità della musica, come il genere, lo stile, la melodia, ecc. 
"""
)

with st.container(border=True):
    draw_blackbox()

st.info(
"""
La **temperatura** è un parametro che controlla la casualità della musica generata. Maggiore è la temperatura, più casuale sarà la musica.
"""
)
