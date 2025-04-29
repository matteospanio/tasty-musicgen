import streamlit as st

st.title("Un po' di teoria")

st.markdown(
"""
## Come impara l'AI?

📊 L’AI ha bisogno di dati per apprendere

- I modelli AI si basano su grandi quantità di dati per identificare pattern e generare output coerenti.
- Maggiore è la qualità e la quantità dei dati, migliore sarà il risultato.

🧠 Un’analogia con l’apprendimento umano

- 👶 Come le persone imparano da esperienze e esercizi, l’AI viene allenata su specifici compiti.
- 🎯 Il modello impara a riconoscere strutture e regole nei dati per generare risultati plausibili.

🔄 Dati in un formato comprensibile

- 💾 L’AI non comprende direttamente la musica o il linguaggio umano.
- 📌 I dati devono essere convertiti in una forma che il modello possa elaborare (es. token, spettrogrammi, embeddings).
"""
)

st.divider()

st.markdown(
"""
## Cosa pensa l'AI?

Perché la rappresentazione è importante?

- I modelli AI lavorano con sequenze di simboli (token).
- La musica deve essere convertita in un formato comprensibile per la rete neurale.

### Principali strategie di rappresentazione
"""
)

col1, col2 = st.columns(2)

with col1:
    st.markdown(
    """
    #### 🎼 MIDI/ABC/Lilypond

    -  🎹 Rappresentazione simbolica della musica.
    -  📍 Note, durate, dinamiche.
    - 📌 Ideale per la generazione strutturata e la modifica di brani.
    """
    )

with col2:
    st.markdown(
        """
    #### 🌊 Spettrogrammi

    - 🔊 Rappresentazione visiva della frequenza nel tempo.
    - 📌 Utile per modelli che processano l’audio come immagini.
    """
    )

