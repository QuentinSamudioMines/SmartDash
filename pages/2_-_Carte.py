"""
# Carte - Visualisation cartographique des scénarios de rénovation énergétique
# 
# Cette page permet de visualiser sur une carte les résultats des simulations
# de rénovation énergétique lancées depuis le tableau de bord principal.
# Elle affiche les consommations par bâtiment selon le scénario et l'année sélectionnés.
#
# Dépendances requises :
# pip install streamlit geopandas folium streamlit-folium branca
"""

# ============================================================================
# IMPORTS ET CONFIGURATION
# ============================================================================

import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
from pathlib import Path
import numpy as np
from branca.colormap import LinearColormap, linear

# Import des modules métier (même logique que le tableau de bord)
from logic.param import annees, scenarios_temporelles
from logic.func import prepare_strategies

# Configuration de la page Streamlit
st.set_page_config(
    page_title="SmartE - Visualisation Cartographique",
    page_icon="images/logo.png",
    layout="wide"
)

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def setup_page_header():
    """Configure l'en-tête de la page avec logo et titre"""
    col_logo, col_title = st.columns([1, 5])
    
    with col_logo:
        st.image("images/logo.png", width=140)
    
    with col_title:
        st.title("Visualisation Cartographique du Scénario")

def check_session_prerequisites():
    """Vérifie que les prérequis de session sont présents"""
    required_keys = ['scenario_name', 'strategy_name', 'usage_selection', 'substitutions', 'city_data']
    missing_keys = [key for key in required_keys if key not in st.session_state]
    
    if missing_keys:
        st.warning(
            "⚠️ Veuillez d'abord lancer une simulation dans le **Tableau de bord** "
            "pour visualiser la carte du scénario."
        )
        st.info(
            "💡 **Comment procéder :**\n"
            "1. Retournez à la page 'Tableau de bord'\n"
            "2. Configurez vos paramètres de simulation\n"
            "3. Attendez que les résultats s'affichent\n"
            "4. Revenez sur cette page pour voir la carte"
        )
        return False
    
    return True

@st.cache_data
def load_geographic_data():
    """Charge et assemble les données géographiques depuis les fichiers PKL"""
    parts = []
    missing_files = []
    
    for i in range(1, 4):
        file_path = Path(f"city_part{i}.pkl")
        if file_path.exists():
            try:
                parts.append(pd.read_pickle(file_path))
            except Exception as e:
                st.error(f"Erreur lors du chargement de {file_path}: {e}")
                return None
        else:
            missing_files.append(f"city_part{i}.pkl")
    
    if missing_files:
        st.error(f"Fichiers géographiques manquants : {', '.join(missing_files)}")
        return None
    
    if not parts:
        st.error("Aucun fichier de données géographiques trouvé")
        return None
    
    # Assemblage et conversion géographique
    try:
        city_df = pd.concat(parts, ignore_index=True)
        city_gdf = gpd.GeoDataFrame(city_df, geometry='geometry')
        city_gdf = city_gdf.set_crs("EPSG:2154", allow_override=True).to_crs("EPSG:4326")
        return city_gdf
    except Exception as e:
        st.error(f"Erreur lors de l'assemblage des données géographiques : {e}")
        return None

def calculate_scenario_consumption(gdf, params, year):
    """Calcule la consommation de chaque bâtiment pour un scénario et une année donnés"""
    try:
        # Préparation des stratégies
        all_strategies = prepare_strategies(gdf)
        gdf_sorted = all_strategies[params["strategy_name"]].copy()
        
        # Récupération du scénario et calcul du taux de rénovation
        scenario_data = scenarios_temporelles[params["scenario_name"]]
        scenario_data = scenario_data * (params['coverage_rate'] / 100.0)
        year_index = list(annees).index(year)
        renovation_percentage = scenario_data[year_index]
        n_reno = int(renovation_percentage * len(gdf_sorted))
        
        # Calcul des consommations actuelles
        gdf_sorted['conso_actuelle'] = gdf_sorted['Consommation par m² par an (en kWh/m².an)_basic']
        
        # Application des rénovations
        if n_reno > 0:
            reno_indices = gdf_sorted.head(n_reno).index
            gdf_sorted.loc[reno_indices, 'conso_actuelle'] = gdf_sorted.loc[
                reno_indices, 'Consommation par m² par an (en kWh/m².an)'
            ]
        
        return gdf_sorted
        
    except Exception as e:
        st.error(f"Erreur lors du calcul du scénario : {e}")
        return gdf

def get_session_parameters():
    """Récupère les paramètres de simulation depuis la session"""
    return {
        'scenario_name': st.session_state['scenario_name'],
        'strategy_name': st.session_state['strategy_name'],
        'usage_selection': st.session_state['usage_selection'], 
        'substitutions': st.session_state.get('substitutions', []),
        'coverage_rate': st.session_state.get('coverage_rate', 30)
    }

def setup_sidebar_controls(params, city_gdf):
    """Configure les contrôles de la barre latérale"""
    st.sidebar.header("🎛️ Contrôles de visualisation")
    
    # === SÉLECTEUR D'ANNÉE ===
    with st.sidebar.expander("📅 Année de simulation", expanded=True):
        selected_year = st.slider(
            "Choisissez une année :",
            min_value=int(annees.min()),
            max_value=int(annees.max()),
            value=2024,
            step=1,
            help="Année pour laquelle visualiser les consommations"
        )
    
    # === SÉLECTEUR DE QUARTIER ===
    with st.sidebar.expander("🏘️Commune", expanded=True):
        # Préparation de la liste des quartiers
        com_codes = city_gdf['NOM_COM'].dropna().astype(str).unique().tolist()
        com_codes_sorted = sorted(com_codes, key=lambda x: str(x).zfill(10))
        com_options = ["Toutes les communes"] + com_codes_sorted
        
        selected_com = st.selectbox(
            "Filtrer par quartier :",
            options=com_options,
            help="Sélectionnez un quartier spécifique ou visualisez tous les bâtiments"
        )
    
    # === AFFICHAGE DES PARAMÈTRES DE SIMULATION ===
    display_simulation_parameters(params, selected_year)
    
    return selected_year, selected_com

def display_simulation_parameters(params, selected_year):
    """Affiche les paramètres de la simulation en cours"""
    with st.sidebar.expander("📋 Paramètres de simulation", expanded=True):
        st.markdown("**Configuration actuelle :**")
        st.markdown(f"- **Taux de rénovation :** `{params['coverage_rate']:.0f}%`")
        st.markdown(f"- **Scénario :** `{params['scenario_name']}`")
        st.markdown(f"- **Stratégie :** `{params['strategy_name']}`")
        st.markdown(f"- **Périmètre :** `{params['usage_selection']}`")
        st.markdown(f"- **Année visualisée :** `{selected_year}`")
        
        # Affichage des conversions énergétiques
        st.markdown("**Conversions énergétiques :**")
        substitutions = params['substitutions']
        if substitutions:
            for source, target, percentage in substitutions:
                st.markdown(f"  - `{source}` → `{target}` ({percentage}%)")
        else:
            st.markdown("  - *Aucune conversion*")
        
        # Calcul et affichage du taux de rénovation pour l'année
        try:
            scenario_data = scenarios_temporelles[params['scenario_name']]
            scenario_data = scenario_data * (params['coverage_rate'] / 100.0)
            year_index = list(annees).index(selected_year)
            renovation_rate = scenario_data[year_index]
            st.markdown(f"- **Taux de rénovation en {selected_year} :** `{renovation_rate:.1%}`")
        except:
            st.markdown("- **Taux de rénovation :** *Calcul impossible*")

def filter_and_prepare_display_data(city_gdf_scenario, selected_commune):
    """Filtre et prépare les données pour l'affichage selon le quartier sélectionné"""
     # Conversion en WGS84 si nécessaire
    if str(city_gdf_scenario.crs) != 'EPSG:4326':
        city_gdf_scenario = city_gdf_scenario.to_crs(epsg=4326)
    if selected_commune == "Toutes les communes":
        display_gdf = city_gdf_scenario
        zoom_level = 12
        map_center = [51.034, 2.376]  # Centre de Dunkerque
    else:
        # Filtrage par quartier IRIS
        display_gdf = city_gdf_scenario[
            city_gdf_scenario['NOM_COM']== selected_commune
        ].copy()
        
        if len(display_gdf) > 0:
            # Calcul du centre du quartier
            try:
                bounds = display_gdf.total_bounds
                map_center = [(bounds[1] + bounds[3])/2, (bounds[0] + bounds[2])/2]
                zoom_level = 14
            except:
                map_center = [51.034, 2.376]
                zoom_level = 12
        else:
            map_center = [51.034, 2.376]
            zoom_level = 12

    return display_gdf, map_center, zoom_level

def optimize_data_for_display(display_gdf, max_buildings=10000):
    """Optimise les données pour l'affichage en limitant le nombre de bâtiments"""
    if len(display_gdf) > max_buildings:
        st.warning(
            f"⚡ Affichage limité à {max_buildings:,} bâtiments "
            f"(sur {len(display_gdf):,}) pour garantir les performances."
        )
        display_gdf = display_gdf.sample(n=max_buildings, random_state=42)
    
    return display_gdf

def create_consumption_map(display_gdf, map_center, zoom_level):
    """Crée la carte Folium avec les consommations par bâtiment"""
    # Création de la carte de base
    m = folium.Map(
        location=map_center, 
        zoom_start=zoom_level, 
        tiles="CartoDB positron"
    )
    
    # Configuration de la palette de couleurs
    max_conso = min(display_gdf['conso_actuelle'].max(), 300)
    min_conso = display_gdf['conso_actuelle'].min()
    
    colormap = LinearColormap(
        colors=['green', 'yellow', 'red'],
        index=[min_conso, (min_conso + max_conso) / 2, max_conso],
        vmin=min_conso,
        vmax=max_conso
    )
    colormap.caption = 'Consommation spécifique (kWh/m².an)'
    m.add_child(colormap)
    
    # Ajout des bâtiments à la carte
    folium.GeoJson(
        display_gdf,
        style_function=lambda feature: {
            'fillColor': colormap(min(feature['properties']['conso_actuelle'], max_conso)),
            'color': colormap(min(feature['properties']['conso_actuelle'], max_conso)),
            'weight': 0.5,
            'fillOpacity': 0.7,
        },
        popup=folium.GeoJsonPopup(
            fields=["UseType", "conso_actuelle", "energie_imope", "CODE_IRIS"],
            aliases=["Usage:", "Conso. (kWh/m²/an):", "Énergie:", "Quartier:"],
            localize=True,
        ),
        tooltip=folium.GeoJsonTooltip(
            fields=["UseType", "conso_actuelle"],
            aliases=["Usage:", "Consommation:"]
        )
    ).add_to(m)
    
    return m

def display_statistics(display_gdf, selected_year):
    """Affiche les statistiques des données visualisées"""
    st.subheader(f"📊 Statistiques - Année {selected_year}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Bâtiments affichés", f"{len(display_gdf):,}")
    
    with col2:
        avg_conso = display_gdf['conso_actuelle'].mean()
        st.metric("Consommation moyenne", f"{avg_conso:.1f} kWh/m²/an")
    
    with col3:
        max_conso = display_gdf['conso_actuelle'].max()
        st.metric("Consommation max", f"{max_conso:.1f} kWh/m²/an")
    
    with col4:
        min_conso = display_gdf['conso_actuelle'].min()
        st.metric("Consommation min", f"{min_conso:.1f} kWh/m²/an")

def display_legend():
    """Affiche la légende de la carte"""
    with st.expander("🎨 Légende de la carte", expanded=False):
        st.markdown("""
        **Couleurs des bâtiments :**
        - 🟡 **Jaune clair** : Faible consommation (< 100 kWh/m²/an)
        - 🟠 **Orange** : Consommation modérée (100-300 kWh/m²/an)  
        - 🔴 **Rouge** : Forte consommation (> 300 kWh/m²/an)
        
        **Interactions :**
        - **Survol** : Affiche l'usage et la consommation
        - **Clic** : Popup avec informations détaillées
        - **Zoom** : Molette de la souris ou boutons +/-
        """)

# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def main():
    """Fonction principale de la page carte"""
    
    # === CONFIGURATION DE LA PAGE ===
    setup_page_header()
    
    # === VÉRIFICATION DES PRÉREQUIS ===
    if not check_session_prerequisites():
        st.stop()
    
    # === RÉCUPÉRATION DES PARAMÈTRES DE SESSION ===
    params = get_session_parameters()
    city_data = st.session_state['city_data']
    
    if city_data is None or len(city_data) == 0:
        st.error("❌ Aucune donnée disponible. Relancez une simulation depuis le tableau de bord.")
        st.stop()
    
    # === CONFIGURATION DES CONTRÔLES ===
    selected_year, selected_commune = setup_sidebar_controls(params, city_data)
    
    # === CALCUL DU SCÉNARIO ===
    st.subheader(f"🗺️ Consommations énergétiques en {selected_year}")
    
    with st.spinner("Calcul des consommations par bâtiment..."):
        city_gdf_scenario = calculate_scenario_consumption(
            city_data, 
            params,
            selected_year
        )
    
    # === PRÉPARATION DES DONNÉES D'AFFICHAGE ===
    display_gdf, map_center, zoom_level = filter_and_prepare_display_data(
        city_gdf_scenario, selected_commune
    )
    
    if len(display_gdf) == 0:
        st.warning(f"⚠️ Aucun bâtiment trouvé pour le quartier sélectionné : {selected_commune}")
        st.stop()
    
    # Optimisation pour l'affichage
    display_gdf = optimize_data_for_display(display_gdf)
    
    # === AFFICHAGE DES STATISTIQUES ===
    display_statistics(display_gdf, selected_year)

    
    # === CRÉATION ET AFFICHAGE DE LA CARTE ===
    st.markdown("---")
    
    with st.spinner("Génération de la carte..."):
        m = create_consumption_map(display_gdf, map_center, zoom_level)
    
    # Affichage de la carte
    st_folium(m, width=1400, height=800, returned_objects=[])
    
    # === LÉGENDE ET INFORMATIONS ===
    display_legend()
    
    # === INFORMATIONS COMPLÉMENTAIRES ===
    st.markdown("---")
    st.info(
        "💡 **Astuce :** Utilisez les contrôles de la barre latérale pour explorer "
        "différentes années et quartiers. Les couleurs indiquent l'intensité "
        "de consommation énergétique par m²."
    )

# ============================================================================
# POINT D'ENTRÉE DE LA PAGE
# ============================================================================

if __name__ == "__main__":
    main()