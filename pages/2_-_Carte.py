import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
from pathlib import Path

st.set_page_config(
    page_title="SmartE - Carte par Quartier",
    page_icon="images/logo.png",
    layout="wide"
)

st.title("Exploration Cartographique par Quartier (IRIS)")

@st.cache_data
def load_data():
    """Charge, assemble et nettoie les données géographiques."""
    parts = []
    for i in range(1, 4):
        file = Path(f"city_part{i}.pkl")
        if file.exists():
            parts.append(pd.read_pickle(file))
        else:
            st.warning(f"Fichier de données {file} introuvable.")
    
    if not parts:
        st.error("Aucun fichier de données n'a pu être chargé.")
        return None

    # Concaténer et s'assurer que c'est un GeoDataFrame
    city_df = pd.concat(parts, ignore_index=True)
    city_gdf = gpd.GeoDataFrame(city_df, geometry='geometry')
    # S'assurer que le CRS est bien défini (WGS84 est standard pour folium)
    city_gdf = city_gdf.set_crs("EPSG:2154", allow_override=True).to_crs("EPSG:4326")
    return city_gdf

# --- Chargement des données ---
city_gdf = load_data()

if city_gdf is None:
    st.stop()

# --- Barre latérale de filtres ---
st.sidebar.header("Filtres d'affichage")

# Créer la liste des IRIS uniques
iris_list = sorted(city_gdf['CODE_IRIS'].unique().tolist())
iris_list.insert(0, "Tous les quartiers (échantillon)")

selected_iris = st.sidebar.selectbox(
    "Choisissez un quartier (CODE_IRIS) :",
    options=iris_list
)

# --- Filtrage des données ---
if selected_iris == "Tous les quartiers (échantillon)":
    # Afficher un échantillon pour la performance
    display_gdf = city_gdf.sample(n=2000, random_state=42)
    zoom_level = 12
    map_center = [51.034, 2.376] # Centre sur Dunkerque
else:
    display_gdf = city_gdf[city_gdf['CODE_IRIS'] == selected_iris]
    # Centrer la carte sur le quartier sélectionné
    try:
        centroid = display_gdf.unary_union.centroid
        map_center = [centroid.y, centroid.x]
        zoom_level = 15
    except Exception:
        map_center = [51.034, 2.376]
        zoom_level = 12

st.info(f"Affichage de {len(display_gdf)} bâtiments pour le quartier : {selected_iris}")

# --- Création et affichage de la carte ---
m = folium.Map(location=map_center, zoom_start=zoom_level, tiles="CartoDB positron")

popup_fields = {
    "UseType": "Usage",
    "Consommation par m² par an (en kWh/m².an)_basic": "Conso. (kWh/m²/an)",
    "energie_imope": "Énergie"
}

folium.GeoJson(
    display_gdf,
    popup=folium.GeoJsonPopup(
        fields=list(popup_fields.keys()),
        aliases=list(popup_fields.values()),
        localize=True,
        sticky=True
    ),
    tooltip=folium.GeoJsonTooltip(fields=["UseType"])
).add_to(m)

st_folium(m, width=1200, height=800, returned_objects=[])