import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration de la page
st.set_page_config(
    page_title="Analyse Globale CUD",
    page_icon="📊",
    layout="wide"
)

# Style personnalisé
st.markdown("""
    <style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .section-title {
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 5px;
        margin-top: 30px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Tableau de Bord d'Analyse Globale")

# Vérification des données de session
required_keys = [
    'city_data_simulated', 
    'conso_par_vecteur', 
    'conso_par_type_et_vecteur',
    'emissions_par_vecteur', 
    'emissions_par_type_et_vecteur'
]

# Vérification plus détaillée des clés manquantes
missing_keys = [key for key in required_keys if key not in st.session_state]

if missing_keys:
    st.error("⚠️ Données de simulation manquantes ou incomplètes")
    st.markdown("""
    ### Prochaines étapes :
    1. Retournez au **Tableau de bord**
    2. Configurez vos paramètres de simulation
    3. Lancez la simulation avec le bouton 🚀
    4. Revenez sur cette page pour voir les résultats détaillés
    """)
    st.stop()

# Vérification que les DataFrames ne sont pas vides
data_checks = {
    'Données de simulation (city_data_simulated)': st.session_state.city_data_simulated,
    'Consommation par vecteur': st.session_state.conso_par_vecteur,
    'Émissions par vecteur': st.session_state.emissions_par_vecteur
}

for name, data in data_checks.items():
    if data is None or (hasattr(data, 'empty') and data.empty):
        st.error(f"⚠️ {name} est vide ou invalide")
        st.warning("Veuillez relancer la simulation avec des paramètres valides.")
        if st.button("🔄 Relancer la simulation", key=f"retry_{name}"):
            st.switch_page("1_-_Tableau_de_bord.py")
        st.stop()

# Chargement des données
df = st.session_state.city_data_simulated
conso_vecteur = st.session_state.conso_par_vecteur
conso_type_vecteur = st.session_state.conso_par_type_et_vecteur
emissions_vecteur = st.session_state.emissions_par_vecteur
emissions_type_vecteur = st.session_state.emissions_par_type_et_vecteur

# Renommage des colonnes pour correspondre aux données disponibles
df = df.rename(columns={
    "Consommation par m² par an (en kWh/m².an)": "conso_m²_an",
    "Consommation annuelle (en MWh/an)": "conso_totale_mwh",
    "surface_habitable": "SURFACE"
})

# ====================
# Statistiques Globales
# ====================
st.header("🔍 Vue d'ensemble", anchor="stats")

# KPI Principaux
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric("Bâtiments", f"{len(df):,}")
    
with kpi2:
    conso_moy = df['conso_m²_an'].mean()
    conso_tot = df['conso_totale_mwh'].sum()  # Déjà en MWh
    st.metric("Consommation moyenne", f"{conso_moy:.1f} kWh/m²/an", f"Total: {conso_tot:,.0f} MWh/an")
    
with kpi3:
    # Estimation des émissions (à ajuster selon votre facteur d'émission)
    facteur_emission_moyen = 0.2  # Exemple: 200g CO2/kWh
    df['emission_kgco2_m²_an'] = df['conso_m²_an'] * facteur_emission_moyen
    emis_moy = df['emission_kgco2_m²_an'].mean()
    emis_tot = (df['conso_totale_mwh'] * facteur_emission_moyen).sum()  # en tonnes
    st.metric("Émissions moyennes", f"{emis_moy:.1f} kgCO₂/m²/an", f"Total: {emis_tot:,.0f} tCO₂/an")
    
with kpi4:
    intensite = (emis_tot * 1000) / (conso_tot + 1e-6)  # gCO2/kWh
    st.metric("Intensité carbone", f"{intensite:.0f} gCO₂/kWh", "Moyenne du parc")

# ====================
# Analyse par commune
# ====================
st.markdown("## 🏘️ Analyse par commune")

# Préparation des données par commune
# Filtrer les lignes où NOM_COM est manquant ou vide
df = df.dropna(subset=['NOM_COM'])
df = df[df['NOM_COM'] != '']

df_commune = df.groupby("NOM_COM").agg({
    'conso_m²_an': 'mean',
    'conso_totale_mwh': 'sum',
    'emission_kgco2_m²_an': 'mean',
    'ID': 'count'
}).reset_index()

# Calcul des émissions totales (en tonnes de CO2)
facteur_emission = 0.2  # 200g CO2/kWh en moyenne
df_commune['emission_totale_tonnes'] = df_commune['conso_totale_mwh'] * facteur_emission

# Sélection de la métrique à afficher
metrique = st.selectbox(
    "Sélectionnez la métrique à visualiser :",
    ["Consommation moyenne (kWh/m²/an)", "Émissions moyennes (kgCO₂/m²/an)", 
     "Consommation totale (MWh/an)", "Émissions totales (tCO₂/an)"],
    index=0
)

# Configuration du graphique en fonction de la métrique sélectionnée
if "moyenne" in metrique.lower():
    if "consommation" in metrique.lower():
        y_col = 'conso_m²_an'
        y_title = "Consommation (kWh/m²/an)"
    else:  # Émissions moyennes
        y_col = 'emission_kgco2_m²_an'
        y_title = "Émissions (kgCO₂/m²/an)"
    color_scale = px.colors.sequential.Blues
else:  # Totaux
    if "consommation" in metrique.lower():
        y_col = 'conso_totale_mwh'
        y_title = "Consommation (MWh/an)"
    else:  # Émissions totales
        y_col = 'emission_totale_tonnes'
        y_title = "Émissions (tCO₂/an)"
    color_scale = px.colors.sequential.Oranges

# Création du graphique
if not df_commune.empty:
    # Trier les données
    df_sorted = df_commune.sort_values(y_col, ascending=False)
    
    # Créer le graphique avec les bons noms de colonnes
    fig_commune = px.bar(
        df_sorted, 
        x='NOM_COM',  # Utiliser le nom de colonne réel
        y=y_col,
        color=y_col,
        color_continuous_scale=color_scale,
        title=f"{metrique} par commune",
        labels={'NOM_COM': 'Commune', y_col: y_title},
        height=500
    )
else:
    st.warning("Aucune donnée disponible pour l'affichage du graphique par commune.")
    fig_commune = None

# Affichage du graphique s'il existe
if fig_commune is not None:
    # Amélioration du style
    fig_commune.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_tickangle=-45,
        coloraxis_showscale=False,
        xaxis_title=None  # Supprime le titre de l'axe x
    )
    st.plotly_chart(fig_commune, use_container_width=True)
else:
    st.warning("Impossible d'afficher le graphique : données insuffisantes.")

# ====================
# Analyse par vecteur énergétique
# ====================
st.markdown("## ⚡ Analyse par vecteur énergétique")

# Calcul des consommations totales par vecteur énergétique
if 'energie_imope' in df.columns:
    # Si on a les données de consommation par bâtiment
    conso_par_vecteur = df.groupby('energie_imope')['conso_totale_mwh'].sum().reset_index()
    
    # Calcul des émissions par vecteur (en tonnes de CO2)
    facteur_emission = 0.2  # 200g CO2/kWh en moyenne
    conso_par_vecteur['emissions_totales'] = conso_par_vecteur['conso_totale_mwh'] * facteur_emission
    
    # Préparation des données pour les graphiques
    vecteurs = conso_par_vecteur['energie_imope'].tolist()
    conso_totale_par_vecteur = conso_par_vecteur['conso_totale_mwh'].round(0).astype(int).tolist()
    emission_totale_par_vecteur = conso_par_vecteur['emissions_totales'].round(0).astype(int).tolist()
else:
    # Fallback si les données ne sont pas disponibles
    st.warning("Les données de consommation par vecteur énergétique ne sont pas disponibles.")
    vecteurs = []
    conso_totale_par_vecteur = []
    emission_totale_par_vecteur = []

# Création des graphiques côte à côte
col1, col2 = st.columns(2)

with col1:
    # Graphique de répartition de la consommation
    fig_conso = px.pie(
        names=vecteurs,
        values=conso_totale_par_vecteur,
        title="Répartition de la consommation par vecteur (MWh)",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    st.plotly_chart(fig_conso, use_container_width=True)

with col2:
    # Graphique de répartition des émissions
    fig_emis = px.pie(
        names=vecteurs,
        values=emission_totale_par_vecteur,
        title="Répartition des émissions par vecteur (tCO₂)",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Pastel2
    )
    st.plotly_chart(fig_emis, use_container_width=True)

# ====================
# Analyse avancée par ville
# ====================


st.markdown("## 📊 Analyse détaillée par ville")
    
# Sélection de la ville pour l'analyse détaillée
villes_disponibles = sorted(df['NOM_COM'].unique())
ville_selectionnee = st.selectbox("Sélectionnez une ville pour une analyse détaillée :", villes_disponibles)
    
# Filtrage des données pour la ville sélectionnée
df_ville = df[df['NOM_COM'] == ville_selectionnee]
    
if not df_ville.empty:
    col1, col2, col3 = st.columns(3)
        
    with col1:
        st.metric("Nombre de bâtiments", len(df_ville))
    with col2:
        st.metric("Consommation moyenne", f"{df_ville['conso_m²_an'].mean():.1f} kWh/m²/an")
    with col3:
        st.metric("Émissions moyennes", f"{df_ville['emission_kgco2_m²_an'].mean():.1f} kgCO₂/m²/an")
        
    # Répartition par type de bâtiment
    if 'UseType' in df_ville.columns:
        st.markdown("### Répartition par type de bâtiment")
        type_batiment = df_ville['UseType'].value_counts().reset_index()
        type_batiment.columns = ['Type de bâtiment', 'Nombre']
            
        fig_type = px.pie(
            type_batiment, 
            names='Type de bâtiment', 
            values='Nombre',
            title=f"Répartition des bâtiments à {ville_selectionnee}",
            hole=0.4
        )
        st.plotly_chart(fig_type, use_container_width=True)
        
    # Top 10 des bâtiments les plus énergivores
    st.markdown("### Top 10 des bâtiments les plus énergivores")
    top_energivores = df_ville.nlargest(10, 'conso_m²_an')[['ID', 'conso_m²_an', 'emission_kgco2_m²_an']]
    st.dataframe(
        top_energivores.style.background_gradient(cmap='YlOrRd', subset=['conso_m²_an', 'emission_kgco2_m²_an']),
        column_config={
            'ID': 'Identifiant',
            'energie_imope': 'Type d\'énergie',
            'conso_m²_an': 'Consommation (kWh/m²/an)',
            'UseType': 'Type de bâtiment',
            'emission_kgco2_m²_an': 'Émissions (kgCO₂/m²/an)'

        },
        use_container_width=True
    )

    # Analyse par vecteur énergétique pour la ville sélectionnée
    st.markdown("### 🔌 Répartition par vecteur énergétique")
        
    if 'energie_imope' in df_ville.columns and not df_ville.empty:
        # Calcul des consommations par vecteur énergétique pour la ville
        conso_ville_vecteur = df_ville.groupby('energie_imope')['conso_totale_mwh'].sum().reset_index()
            
        # Calcul des émissions (en tonnes de CO2)
        facteur_emission = 0.2  # 200g CO2/kWh en moyenne
        conso_ville_vecteur['emissions_tonnes'] = conso_ville_vecteur['conso_totale_mwh'] * facteur_emission
            
        # Création des onglets pour les graphiques
        tab1, tab2 = st.tabs(["Consommation", "Émissions"])
            
        with tab1:
            # Graphique en camembert de la consommation
            fig_conso = px.pie(
                conso_ville_vecteur,
                values='conso_totale_mwh',
                names='energie_imope',
                title=f'Répartition de la consommation énergétique à {ville_selectionnee}',
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig_conso, use_container_width=True)
                
        with tab2:
            # Graphique en barres des émissions
            fig_emis = px.bar(
                conso_ville_vecteur.sort_values('emissions_tonnes', ascending=False),
                x='energie_imope',
                y='emissions_tonnes',
                title=f'Émissions par vecteur énergétique à {ville_selectionnee}',
                color='emissions_tonnes',
                color_continuous_scale=px.colors.sequential.Reds,
                labels={'energie_imope': 'Vecteur énergétique', 'emissions_tonnes': 'Émissions (tCO₂/an)'}
            )
            st.plotly_chart(fig_emis, use_container_width=True)
                
        # Tableau détaillé
        st.markdown("#### Détails par vecteur énergétique")
            
        # Calcul des pourcentages
        total_conso = conso_ville_vecteur['conso_totale_mwh'].sum()
        total_emis = conso_ville_vecteur['emissions_tonnes'].sum()
            
        # Création du tableau de données
        details_df = conso_ville_vecteur.sort_values('conso_totale_mwh', ascending=False)
        details_df['% Consommation'] = (details_df['conso_totale_mwh'] / total_conso * 100).round(1)
        details_df['% Émissions'] = (details_df['emissions_tonnes'] / total_emis * 100).round(1)
            
        # Formatage des colonnes
        details_display = details_df.rename(columns={
            'energie_imope': 'Vecteur énergétique',
            'conso_totale_mwh': 'Consommation (MWh/an)',
            'emissions_tonnes': 'Émissions (tCO₂/an)'
        })
            
        # Affichage du tableau
        st.dataframe(
            details_display[[
                'Vecteur énergétique',
                'Consommation (MWh/an)',
                '% Consommation',
                'Émissions (tCO₂/an)',
                '% Émissions'
            ]].style.format({
                'Consommation (MWh/an)': '{:,.0f}',
                'Émissions (tCO₂/an)': '{:.1f}'
            }),
            use_container_width=True,
            height=min(400, 35 * len(details_df) + 35)  # Hauteur dynamique
        )
    else:
        st.info("Aucune donnée de vecteur énergétique disponible pour cette ville.")
else:
    st.warning("Impossible d'afficher le graphique : données insuffisantes.")

    # Filtrage des données pour la ville sélectionnée
    df_ville = df[df['NOM_COM'] == ville_selectionnee]
    
    if not df_ville.empty:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Nombre de bâtiments", len(df_ville))
        with col2:
            st.metric("Consommation moyenne", f"{df_ville['conso_m²_an'].mean():.1f} kWh/m²/an")
        with col3:
            st.metric("Émissions moyennes", f"{df_ville['emission_kgco2_m²_an'].mean():.1f} kgCO₂/m²/an")
        
        # Répartition par type de bâtiment
        if 'UseType' in df_ville.columns:
            st.markdown("### Répartition par type de bâtiment")
            type_batiment = df_ville['UseType'].value_counts().reset_index()
            type_batiment.columns = ['Type de bâtiment', 'Nombre']
            
            fig_type = px.pie(
                type_batiment, 
                names='Type de bâtiment', 
                values='Nombre',
                title=f"Répartition des bâtiments à {ville_selectionnee}",
                hole=0.4
            )
            st.plotly_chart(fig_type, use_container_width=True)
        
        # Top 10 des bâtiments les plus énergivores
        st.markdown("### Top 10 des bâtiments les plus énergivores")
        top_energivores = df_ville.nlargest(10, 'conso_m²_an')[['ID', 'conso_m²_an', 'emission_kgco2_m²_an']]
        st.dataframe(
            top_energivores.style.background_gradient(cmap='YlOrRd', subset=['conso_m²_an', 'emission_kgco2_m²_an']),
            column_config={
                'ID': 'Identifiant',
                'conso_m²_an': 'Consommation (kWh/m²/an)',
                'emission_kgco2_m²_an': 'Émissions (kgCO₂/m²/an)'
            },
            use_container_width=True
        )

# ====================
# Analyse des plans de rénovation
# ====================
st.markdown("## 🏗️ Analyse des plans de rénovation")

# Simulation des économies potentielles
st.markdown("### Impact potentiel des rénovations")

# Récupération des paramètres de simulation depuis la session
if 'simulation_params' not in st.session_state:
    st.session_state.simulation_params = {
        'taux_residentiel': 30,  # Valeurs par défaut
        'taux_tertiaire': 20,
        'strategie': 'Par consommation/m²',
        'scenario_temporel': 'Linéaire'
    }

# Affichage des paramètres dans la barre latérale
with st.sidebar.expander("📋 Paramètres de simulation", expanded=True):
    st.markdown("**Configuration actuelle :**")
    st.markdown(f"- **Taux de rénovation résidentiel :** `{st.session_state.simulation_params['taux_residentiel']}%`")
    st.markdown(f"- **Taux de rénovation tertiaire :** `{st.session_state.simulation_params['taux_tertiaire']}%`")
    st.markdown(f"- **Stratégie :** `{st.session_state.simulation_params['strategie']}`")
    st.markdown(f"- **Scénario temporel :** `{st.session_state.simulation_params['scenario_temporel']}`")
    
    # Calcul du taux de rénovation moyen pondéré
    try:
        total_buildings = len(df)
        if total_buildings > 0:
            residential_count = len(df[df['UseType'].str.contains('Résidentiel', na=False)])
            tertiary_count = total_buildings - residential_count
            
            avg_renovation_rate = (
                (residential_count * st.session_state.simulation_params['taux_residentiel']) + 
                (tertiary_count * st.session_state.simulation_params['taux_tertiaire'])
            ) / total_buildings
            
            st.markdown(f"- **Taux de rénovation moyen :** `{avg_renovation_rate:.1f}%`")
    except:
        pass

# Utilisation des paramètres de la session
taux_residentiel = st.session_state.simulation_params['taux_residentiel']
taux_tertiaire = st.session_state.simulation_params['taux_tertiaire']
strategie = st.session_state.simulation_params['strategie']
scenario_temporel = st.session_state.simulation_params['scenario_temporel']

# Durée du plan de rénovation (fixe ou paramétrable)
duree_plan = 15  # Jusqu'en 2050 - 2024 = 26 ans, mais on garde 15 pour la cohérence avec l'UI

if 'conso_totale_mwh' in df.columns and 'NOM_COM' in df.columns and 'UseType' in df.columns:
    # Calcul des économies potentielles par ville et par type de bâtiment
    df['taux_renovation'] = df['UseType'].apply(
        lambda x: taux_residentiel/100 if 'Résidentiel' in str(x) else taux_tertiaire/100
    )
    
    # Calcul des économies potentielles en fonction du type de bâtiment
    df['economie_potentielle'] = df['conso_totale_mwh'] * df['taux_renovation']
    
    # Agrégation par ville
    economies_ville = df.groupby('NOM_COM').agg({
        'economie_potentielle': 'sum',
        'conso_totale_mwh': 'sum'
    }).reset_index()
    
    # Calcul du pourcentage d'économies
    economies_ville['% Économie potentielle'] = (economies_ville['economie_potentielle'] / 
                                              economies_ville['conso_totale_mwh']) * 100
    
    # Renommage des colonnes
    economies_ville.columns = ['Ville', 'Économie annuelle (MWh)', 'Consommation totale (MWh)', '% Économie potentielle']
    
    # Calcul des économies sur la durée du plan
    economies_ville['Économie sur la durée du plan (MWh)'] = economies_ville['Économie annuelle (MWh)'] * duree_plan
    
    # Affichage des résultats
    st.markdown(f"#### Économies potentielles avec les taux de rénovation sélectionnés")
    st.markdown(f"- **Résidentiel** : {taux_residentiel}% des bâtiments")
    st.markdown(f"- **Tertiaire** : {taux_tertiaire}% des bâtiments")
    
    # Top 5 des villes avec le plus grand potentiel d'économies
    st.markdown("##### Top 5 des villes avec le plus grand potentiel d'économies")
    top_villes = economies_ville.nlargest(5, 'Économie annuelle (MWh)')
    fig_economies = px.bar(
        top_villes,
        x='Ville',
        y='Économie annuelle (MWh)',
        color='Économie annuelle (MWh)',
        title=f"Économies annuelles potentielles par ville (Top 5)",
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig_economies, use_container_width=True)
    
    # Détail des économies
    st.markdown("##### Détail par ville")
    st.dataframe(
        economies_ville.sort_values('Économie annuelle (MWh)', ascending=False),
        column_config={
            'Ville': 'Ville',
            'Économie annuelle (MWh)': st.column_config.NumberColumn(
                'Économie annuelle (MWh)',
                format="%.0f MWh"
            ),
            'Économie sur la durée du plan (MWh)': st.column_config.NumberColumn(
                f'Économie sur {duree_plan} ans (MWh)',
                format="%.0f MWh"
            )
        },
        use_container_width=True,
        hide_index=True
    )

# Recommandations de rénovation
st.markdown("### Recommandations de rénovation")

recommandations = [
    "🔹 Isolation des combles et des murs pour réduire les déperditions thermiques",
    "🔹 Remplacement des fenêtres par du double ou triple vitrage",
    "🔹 Installation de systèmes de chauffage à haute efficacité énergétique",
    "🔹 Mise en place de systèmes de régulation et de programmation du chauffage",
    "🔹 Récupération des eaux pluviales pour les usages non sanitaires",
    "🔹 Installation de panneaux solaires pour la production d'eau chaude sanitaire"
]

for reco in recommandations:
    st.markdown(f"- {reco}")

# ====================
# Analyse par type de bâtiment
# ====================
st.markdown("## 🏢 Analyse par type de bâtiment")

# Préparation des données par type de bâtiment
df_type = df.groupby('UseType').agg({
    'conso_m²_an': 'mean',
    'emission_kgco2_m²_an': 'mean',
    'SURFACE': 'mean',
    'ID': 'count'
}).reset_index()

df_type = df_type.rename(columns={
    'ID': 'Nb_batiments',
    'SURFACE': 'Surface_moyenne',
    'conso_m²_an': 'Conso_moy',
    'emission_kgco2_m²_an': 'Emission_moy'
})

# Création des graphiques
col1, col2 = st.columns(2)

with col1:
    # Graphique de la consommation moyenne par type
    fig_conso_type = px.bar(
        df_type.sort_values('Conso_moy', ascending=False),
        x='UseType',
        y='Conso_moy',
        title='Consommation moyenne par type de bâtiment',
        labels={'Conso_moy': 'Consommation (kWh/m²/an)', 'UseType': 'Type de bâtiment'},
        color='Conso_moy',
        color_continuous_scale=px.colors.sequential.Blues
    )
    st.plotly_chart(fig_conso_type, use_container_width=True)

with col2:
    # Graphique des émissions moyennes par type
    fig_emis_type = px.bar(
        df_type.sort_values('Emission_moy', ascending=False),
        x='UseType',
        y='Emission_moy',
        title='Émissions moyennes par type de bâtiment',
        labels={'Emission_moy': 'Émissions (kgCO₂/m²/an)', 'UseType': 'Type de bâtiment'},
        color='Emission_moy',
        color_continuous_scale=px.colors.sequential.Oranges
    )
    st.plotly_chart(fig_emis_type, use_container_width=True)

# Analyse par type de bâtiment
st.markdown("### 🔍 Analyse par type de bâtiment")

df_type = df.groupby('UseType').agg({
    'conso_m²_an': 'mean',
    'emission_kgco2_m²_an': 'mean',
    'ID': 'count',
    'SURFACE': 'mean'
}).reset_index()

fig_type = px.bar(
    df_type.sort_values('conso_m²_an', ascending=False),
    x='UseType',
    y=['conso_m²_an', 'emission_kgco2_m²_an'],
    barmode='group',
    title='Performance énergétique par type de bâtiment',
    labels={
        'value': 'Valeur',
        'UseType': 'Type de bâtiment',
        'variable': 'Métrique'
    },
    color_discrete_map={
        'conso_m²_an': '#1f77b4',
        'emission_kgco2_m²_an': '#ff7f0e'
    }
)

# Personnalisation du graphique
fig_type.update_layout(
    yaxis_title='Valeur',
    legend_title='Métrique',
    xaxis_tickangle=-45,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)'
)

# Mise à jour des noms des séries dans la légende
fig_type.for_each_trace(lambda t: t.update(name='Consommation (kWh/m²/an)' 
                                        if t.name == 'conso_m²_an' 
                                        else 'Émissions (kgCO₂/m²/an)'))

st.plotly_chart(fig_type, use_container_width=True)

# ====================
# Export des données
# ====================
st.markdown("## 📊 Export des données")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Données brutes")
    st.download_button(
        "💾 Télécharger les données brutes (CSV)",
        data=df.to_csv(index=False, sep=';', decimal=',').encode('utf-8-sig'),
        file_name="donnees_brutes_CUD.csv",
        mime='text/csv',
        help="Téléchargez l'ensemble des données brutes au format CSV"
    )

with col2:
    st.markdown("### Synthèse des résultats")
    st.download_button(
        "📊 Télécharger la synthèse (CSV)",
        data=df.groupby(['NOM_COM', 'UseType']).agg({
            'conso_m²_an': 'mean',
            'emission_kgco2_m²_an': 'mean',
            'ID': 'count',
            'SURFACE': 'sum',
            'conso_totale_mwh': 'sum'
        }).reset_index().to_csv(index=False, sep=';', decimal=',').encode('utf-8-sig'),
        file_name="synthese_resultats_CUD.csv",
        mime='text/csv',
        help="Téléchargez une synthèse des résultats par commune, type de bâtiment et période de construction"
    )

# Ajout d'un expander avec des informations supplémentaires
with st.expander("ℹ️ À propos de ces données"):
    st.markdown("""
    **Source des données :** Modèle de simulation énergétique de la CUD
    
    **Période d'analyse :** Données annuelles
    
    **Définitions :**
    - **Consommation énergétique :** Exprimée en kWh/m²/an
    - **Émissions de CO₂ :** Exprimées en kgCO₂/m²/an
    - **Intensité carbone :** Exprimée en gCO₂/kWh
    
    **Note :** Les valeurs sont des estimations basées sur les hypothèses du modèle de simulation.
    """)