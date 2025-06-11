import streamlit as st
import os
from pathlib import Path
import streamlit.components.v1 as components

st.set_page_config(
    page_title="SmartE - MAP 2050 Project",
    page_icon="images/logo.png",
    layout="wide"
)

# Titre principal
st.title("Cartes détaillées du projet MAP 2050")
# Dossier contenant les fichiers .html
maps_dir = Path("maps")
if not maps_dir.exists():
    st.error(f"Répertoire {maps_dir} introuvable.")
else:
    # Liste des fichiers .html
    map_files = sorted(maps_dir.glob("*.html"))
    if not map_files:
        st.warning("Aucun fichier .html trouvé dans le répertoire maps_html.")
    else:
        # Définition de la légende des zones
        legend = {
            'Zone 0': '#a50026',
            'Zone 1': '#b10b26',
            'Zone 2': '#feca79',
            'Zone 3': '#fee491',
            'Zone 4': '#e0f295',
            'Zone 5': '#fec877',
            'Zone 6': '#fff7b2',
            'Zone 8': '#fece7c',
            'Zone 10': '#006837',
        }
        st.subheader("Légende des couleurs :")
        for zone, color in legend.items():
            st.markdown(f"- <span style='color:{color}'>■</span> **{zone}**", unsafe_allow_html=True)
        st.markdown("---")
        for map_file in map_files:
            st.subheader(map_file.stem)
            # Chargement du fichier HTML dans un iframe
            html_content = map_file.read_text(encoding='utf-8')
            components.html(html_content, height=600, scrolling=True)

# Pied de page
st.markdown("---")