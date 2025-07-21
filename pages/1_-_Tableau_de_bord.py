# ============================================================================
# IMPORTS ET CONFIGURATION
# ============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import warnings

# Import des modules métier personnalisés
from logic.param import (
    annees, 
    electricity_carbone_factor, 
    facteurs_carbone, 
    scenarios_temporelles
)
from logic.func import (
    calculate_energy_profile_by_sector,
    calculate_heating_efficiencies,
    create_consumption_chart,
    create_cumulative_emissions_chart,
    create_dynamic_histogram,
    create_emissions_chart,
    filter_data_by_selection,
    load_sample_data,
    prepare_strategies,
    simulate,
    synthesize_results
)

# Suppression des avertissements Streamlit non critiques
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")

# Configuration de la page Streamlit
st.set_page_config(
    page_title="SmartE - Simulation Rénovation Énergétique",
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
        st.title("SmartE - Étude de la transition énergétique du parc immobilier de la CUD")

def display_introduction():
    """Affiche la bannière d'introduction et les instructions d'utilisation"""
    # Bannière d'introduction
    st.markdown("""
    <h3> Optimisez votre stratégie de rénovation énergétique</h3>
    <p>Cet outil permet de simuler différents scénarios de rénovation du parc immobilier et d'analyser leur impact sur :</p>
    <ul>
        <li>Les <b>consommations énergétiques</b> (MWh)</li>
        <li>Les <b>émissions de CO₂</b> (tonnes équivalent carbone)</li>
        <li>La <b>transition énergétique</b> du territoire</li>
    </ul>
    <p>Période d'analyse : <b>2024 à 2050</b> (objectif neutralité carbone)</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Section "Comment ça marche" - repliable
    with st.expander("ℹ️ Comment utiliser ce simulateur", expanded=False):
        st.markdown("""
        1. **Sélectionnez** le périmètre d'étude (résidentiel seul ou avec tertiaire)
        2. **Paramétrez** votre simulation dans la barre latérale ←
        3. **Visualisez** les résultats en quelques secondes
        4. **Comparez** différents scénarios
        
        ### Concepts clés :
        - **Périmètre d'étude** : Type de bâtiments inclus dans l'analyse
        - **Stratégies de rénovation** : Ordre de priorité des bâtiments à rénover  
        - **Scénarios temporels** : Rythme de déploiement des rénovations  
        - **Conversion énergétique** : Substitution entre sources d'énergie  
        - **Facteurs carbone** : Intensité CO₂ de chaque énergie (kgCO₂/kWh)
        """)

def initialize_data():
    """Charge et initialise les données de la ville si pas déjà en mémoire"""
    with st.spinner("Chargement des données..."):
        st.session_state.city_data = load_sample_data()
    
    # Sauvegarde des scénarios pour utilisation dans d'autres pages
    st.session_state['scenarios_temporelles'] = scenarios_temporelles

    
    return st.session_state.city_data

def setup_study_perimeter_selection():
    """Configure la sélection du périmètre d'étude dans la sidebar"""
    st.sidebar.markdown("### 🏠 Périmètre d'étude")
    
    # Sélection du type d'usage (résidentiel/tertiaire)
    with st.sidebar.expander("📋 Type de bâtiments", expanded=False):
        st.markdown("""
        **Choisissez quels types de bâtiments inclure dans l'analyse :**
        - 🏠 **Résidentiel** : Logements
        - 🏢 **Tertiaire** : Bâtiments d'activité (bureaux, commerces, etc.)
        - 🏠+🏢 **Résidentiel + Tertiaire** 
        """)
        
        usage_selection = st.radio(
            "Secteurs à analyser",
            options=["Résidentiel", "Tertiaire", "Résidentiel + Tertiaire"],
            index=2,  # Par défaut : Résidentiel + Tertiaire
            help="Définit le périmètre des bâtiments inclus dans l'analyse"
        )
    
    # Sélection géographique CUD (Communauté Urbaine de Dunkerque)
    with st.sidebar.expander("🏛️ Périmètre géographique", expanded=False):
        st.markdown("""
        **Filtrez sur les bâtiments de la Communauté Urbaine de Dunkerque :**
        - ✅ **Bâtiments CUD uniquement** : Focus sur le patrimoine CUD
        - 🌍 **Tous les bâtiments** : Analyse complète du territoire
        """)
        
        cud_only = st.checkbox(
            "Limiter aux bâtiments CUD",
            value=False,
            help="Si coché, seuls les bâtiments identifiés comme appartenant à la CUD seront analysés"
        )
    
    return usage_selection, cud_only

def display_filtered_data_stats(city_data, usage_selection, cud_only):
    """Affiche les statistiques des données filtrées dans la sidebar"""
    # Calcul des statistiques
    residential_count = len(city_data[city_data["UseType"] == "LOGEMENT"])
    tertiary_count = len(city_data[city_data["UseType"] != "LOGEMENT"])
    total_buildings = len(city_data)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Données filtrées")
    
    # Affichage des métriques
    st.sidebar.metric("Total bâtiments", total_buildings)
    
    # Détail par secteur selon la sélection
    if usage_selection == "Résidentiel + Tertiaire":
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("🏠 Résidentiel", residential_count)
        with col2:
            st.metric("🏢 Tertiaire", tertiary_count)
    elif usage_selection == "Résidentiel":
        st.sidebar.metric("🏠 Résidentiel", residential_count)
    else:
        st.sidebar.metric("🏢 Tertiaire", tertiary_count)
    
    # Information sur le filtre CUD
    if cud_only:
        if 'is_cud' in city_data.columns:
            cud_buildings = len(city_data[city_data['is_cud'] == True])
            st.sidebar.info(f"🏛️ Analyse limitée aux {cud_buildings} bâtiments CUD")
        else:
            st.sidebar.warning("⚠️ Filtre CUD non disponible - Tous les bâtiments inclus")
    
    # Calcul et affichage de la consommation totale
    original_profile = calculate_energy_profile_by_sector(city_data)
    total_consumption = sum([p['consommation_basic'] for p in original_profile.values()])
    st.sidebar.metric("Consommation totale", f"{total_consumption:.0f} MWh/an")

def setup_simulation_parameters():
    """Configure les paramètres de simulation dans la sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.header("🔧 Paramètres de simulation")

    # === Taux de rénovation par usage agrégé ===
    with st.sidebar.expander("🏗️ Taux de rénovation par type global", expanded=False):
        st.markdown("""
        **Définit la proportion du parc rénovée d’ici 2050 pour chaque grand usage.**
        - Résidentiel : logements
        - Tertiaire : écoles, bureaux, commerces, etc.
        """)
        taux_residentiel = st.slider(
            "Résidentiel",
            min_value=0,
            max_value=100,
            value=30,
            step=5,
            help="Taux de rénovation pour les bâtiments résidentiels"
        )

        taux_tertiaire = st.slider(
            "Tertiaire",
            min_value=0,
            max_value=100,
            value=20,
            step=5,
            help="Taux de rénovation pour les bâtiments tertiaires"
        )

    return {
        "Résidentiel": taux_residentiel,
        "Tertiaire": taux_tertiaire
    }


def setup_renovation_strategy_selection(strategies):
    """Configure la sélection de stratégie de rénovation"""
    with st.sidebar.expander("📌 Stratégie de rénovation", expanded=False):
        st.markdown("""
        **Détermine l'ordre dans lequel les bâtiments seront rénovés:**
        - 🔝 Par consommation/m²: Priorité aux bâtiments les plus énergivores
        - 🏢 Par consommation totale: Priorité aux grands consommateurs
        - 🎲 Aléatoire: Sélection non ordonnée
        """)
        selected_strategy = st.selectbox(
            "Stratégie appliquée",
            list(strategies.keys()),
            index=0,
            help="Ordre de priorité pour la rénovation des bâtiments"
        )
    
    return selected_strategy

def setup_temporal_scenario_selection():
    """Configure la sélection du scénario temporel"""
    with st.sidebar.expander("⏱️ Scénario temporel", expanded=False):
        st.markdown("""
        **Définit la vitesse de déploiement des rénovations:**
        - 🏁 Linéaire: Progression constante
        - 🚀 Rapide début: Effort important dès le début
        - 🐢 Lent début: Démarrage progressif
        """)
        selected_scenario = st.selectbox(
            "Profil de déploiement",
            list(scenarios_temporelles.keys()),
            index=0,
            help="Rythme de déploiement des rénovations sur la période"
        )
    
    return selected_scenario

def setup_energy_substitution(original_profile, heating_efficiency_map):
    """Configure le système de substitution des énergies dans la sidebar"""
    with st.sidebar.expander("🔄 Conversion des énergies de chauffage", expanded=False):
        st.markdown("""
        **Remplacez progressivement une source d'énergie par une autre**  
        *Ex: Conversion du fioul vers des pompes à chaleur électriques*
        """)

        # Style CSS pour améliorer l'affichage
        st.markdown("""
        <style>
        .energy-substitution {
            padding: 15px;
            background-color: #f0f2f6;
            border-radius: 10px;
            margin-bottom: 15px;
        }
        </style>
        """, unsafe_allow_html=True)

        # Interface de sélection de la substitution
        with st.container():
            source = st.selectbox(
                "Source à remplacer",
                options=list(original_profile.keys()),
                index=1,  # Gaz par défaut
                key="sub_source",
                help="Énergie actuellement utilisée que vous souhaitez réduire"
            )
            target = st.selectbox(
                "Énergie de remplacement",
                options=[e for e in heating_efficiency_map.keys() if e != source],
                index=0,  # Électricité par défaut
                key="sub_target",
                help="Nouvelle énergie qui remplacera la source"
            )
            rate = st.slider(
                "% de conversion",
                min_value=0,
                max_value=100,
                value=30,
                step=5,
                key="sub_rate",
                help="Pourcentage de conversion maximum d'ici 2050"
            )

        # Gestion des substitutions en session
        if 'substitutions' not in st.session_state:
            st.session_state['substitutions'] = []

        # Bouton d'ajout de substitution
        if st.button("➕ Ajouter cette conversion", use_container_width=True):
            if rate > 0:
                st.session_state['substitutions'].append((source, target, rate))
                st.success(f"Conversion ajoutée : {rate}% de {source} → {target}")
            else:
                st.warning("Veuillez spécifier un pourcentage de conversion")

        # Affichage et gestion des conversions actives
        if st.session_state['substitutions']:
            st.markdown("**Conversions actives :**")
            for i, (src, tgt, pct) in enumerate(st.session_state['substitutions']):
                cols = st.columns([4, 1])
                with cols[0]:
                    st.markdown(f"`{src}` → `{tgt}` ({pct}%)")
                with cols[1]:
                    if st.button("✕", key=f"del_{i}"):
                        st.session_state['substitutions'].pop(i)
                        st.rerun()
    
    return st.session_state['substitutions']

def display_results(conso_par_vecteur, 
                   emissions_par_vecteur, df_selected, annees, scenario_data):
    """Affiche les résultats de la simulation"""
    st.subheader(f"Résultats pour la stratégie: {st.session_state['strategy_name']} - Scénario: {st.session_state['scenario_name']}")
    
    # === GRAPHIQUES PRINCIPAUX ===
    st.subheader("Consommation et émissions annuelles")
    col1, col2 = st.columns(2)
    
    with col1:
        fig_conso = create_consumption_chart(annees, conso_par_vecteur)
        st.plotly_chart(fig_conso, use_container_width=True)
    
    with col2:
        fig_emissions = create_emissions_chart(annees, emissions_par_vecteur)
        st.plotly_chart(fig_emissions, use_container_width=True)
    
    # === NOUVEAU GRAPHIQUE DES ÉMISSIONS CUMULÉES ===
    st.subheader("Impact cumulé des rénovations sur les émissions")
    fig_cumulative = create_cumulative_emissions_chart(annees, emissions_par_vecteur, st.session_state['scenario_name'])
    st.plotly_chart(fig_cumulative, use_container_width=True)
    
    # Explication de l'interprétation
    st.info("""
    **Comment interpréter ce graphique :**
    - La pente de la courbe représente le rythme des émissions
    - Plus la courbe s'aplatit tôt, plus les réductions d'émissions sont précoces
    - Comparer différents scénarios pour voir l'impact du rythme de rénovation
    """)
    
    # === DISTRIBUTION DES CONSOMMATIONS ===
    st.subheader("Distribution des consommations de chauffage du parc")
    fig_distribution = create_dynamic_histogram(df_selected, scenario_data)
    st.plotly_chart(fig_distribution, use_container_width=True)

def display_summary_metrics(city_data, conso_par_vecteur, emissions_par_vecteur):
    """Affiche le bilan énergétique et carbone avec répartition par type de bâtiment et vecteur énergétique"""
    # Calculate comprehensive results
    bilan_stats = synthesize_results(city_data, conso_par_vecteur, emissions_par_vecteur)
    
    st.subheader("📊 Bilan énergétique et carbone (2024-2050)")
    
    # === SECTION 1: KEY PERFORMANCE INDICATORS ===
    st.markdown("### 🎯 Indicateurs clés de performance")
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    total_conso = sum([sum(conso) for conso in conso_par_vecteur.values()])
    total_emissions = sum([sum(emissions) for emissions in emissions_par_vecteur.values()])
    with kpi1:
        st.metric(
            "Consommation totale",
            f"{total_conso:,.0f} MWh",
            help="Somme des consommations énergétiques 2024-2050"
        )
    
    with kpi2:
        st.metric(
            "Émissions totales",
            f"{total_emissions:,.0f} tCO₂",
            help="Somme des émissions carbone 2024-2050"
        )
    
    with kpi3:
        st.metric(
            "Réduction consommation",
            f"{bilan_stats['Réduction conso (%)']:.1f}%",
            delta=f"-{bilan_stats['Réduction conso (MWh)']:,.0f} MWh",
            help="Réduction totale 2024-2050"
        )
    
    with kpi4:
        st.metric(
            "Réduction émissions",
            f"{bilan_stats['Réduction émissions (%)']:.1f}%",
            delta=f"-{bilan_stats['Réduction émissions (tCO₂)']:,.0f} tCO₂",
            help="Réduction totale 2024-2050"
        )
    
    # === SECTION 2: YEARLY COMPARISON ===
    st.markdown("### 📅 Comparatif annuel 2024 vs 2050")
    
    comp1, comp2, comp3 = st.columns(3)
    
    with comp1:
        st.markdown("**Consommation énergétique**")
        st.metric(
            "2024", 
            f"{bilan_stats['Consommation 2024 (MWh)']:,.0f} MWh"
        )
        st.metric(
            "2050",
            f"{bilan_stats['Consommation 2050 (MWh)']:,.0f} MWh",
            delta=f"-{bilan_stats['Réduction conso (MWh)']:,.0f} MWh ({bilan_stats['Réduction conso (%)']:.1f}%)"
        )
    
    with comp2:
        st.markdown("**Émissions carbone**")
        st.metric(
            "2024",
            f"{bilan_stats['Émissions 2024 (tCO₂)']:,.0f} tCO₂"
        )
        st.metric(
            "2050",
            f"{bilan_stats['Émissions 2050 (tCO₂)']:,.0f} tCO₂",
            delta=f"-{bilan_stats['Émissions 2024 (tCO₂)'] - bilan_stats['Émissions 2050 (tCO₂)']:,.0f} tCO₂ ({bilan_stats['Réduction émissions (%)']:.1f}%)"
        )
    
    with comp3:
        st.markdown("**Intensité carbone**")
        st.metric(
            "2024",
            f"{bilan_stats['Intensité carbone 2024 (kgCO₂/MWh)']:.1f} kg/MWh"
        )
        st.metric(
            "2050",
            f"{bilan_stats['Intensité carbone 2050 (kgCO₂/MWh)']:.1f} kg/MWh",
            delta=f"-{bilan_stats['Intensité carbone 2024 (kgCO₂/MWh)'] - bilan_stats['Intensité carbone 2050 (kgCO₂/MWh)']:.1f} kg/MWh"
        )
    
    # === SECTION 3: ENERGY VECTOR BREAKDOWN ===
    st.markdown("### ⚡ Répartition par vecteur énergétique")
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Consommation", "Émissions"])
    
    with tab1:
        st.markdown("**Évolution des consommations par énergie**")
        
        # Prepare data for visualization
        energy_df = pd.DataFrame({
            '2024': [bilan_stats["energy_breakdown"]["by_energy"]["conso_2024"][e] for e in bilan_stats["energy_vectors"]],
            '2050': [bilan_stats["energy_breakdown"]["by_energy"]["conso_2050"][e] for e in bilan_stats["energy_vectors"]],
        }, index=bilan_stats["energy_vectors"])
        
        # Show bar chart
        fig = px.bar(energy_df, barmode='group', 
                    labels={'value': 'Consommation (MWh)', 'variable': 'Année'},
                    title="Consommation par type d'énergie")
        st.plotly_chart(fig, use_container_width=True)
        
        # Show detailed table
        energy_df['Réduction'] = energy_df['2024'] - energy_df['2050']
        energy_df['Réduction %'] = (energy_df['Réduction'] / energy_df['2024']) * 100
        st.dataframe(energy_df.style.format("{:,.0f}"), use_container_width=True)
    
    with tab2:
        st.markdown("**Évolution des émissions par énergie**")
        
        # Prepare data for visualization
        em_df = pd.DataFrame({
            '2024': [bilan_stats["energy_breakdown"]["by_energy"]["emissions_2024"][e] for e in bilan_stats["energy_vectors"]],
            '2050': [bilan_stats["energy_breakdown"]["by_energy"]["emissions_2050"][e] for e in bilan_stats["energy_vectors"]],
        }, index=bilan_stats["energy_vectors"])
        
        # Show bar chart
        fig = px.bar(em_df, barmode='group', 
                    labels={'value': 'Émissions (tCO₂)', 'variable': 'Année'},
                    title="Émissions par type d'énergie")
        st.plotly_chart(fig, use_container_width=True)
        
        # Show detailed table
        em_df['Réduction'] = em_df['2024'] - em_df['2050']
        em_df['Réduction %'] = (em_df['Réduction'] / em_df['2024']) * 100
        st.dataframe(em_df.style.format("{:,.0f}"), use_container_width=True)
    
    # === SECTION 4: BUILDING TYPE BREAKDOWN (if available) ===
    if bilan_stats["building_breakdown"]:
        st.markdown("### 🏢 Répartition par type de bâtiment")
        
        # Prepare data
        building_types = {
            'residential': 'Résidentiel',
            'tertiary': 'Tertiaire'
        }
        
        bld_data = []
        for b_type, b_name in building_types.items():
            bld_data.append({
                'Type': b_name,
                'Consommation 2024 (MWh)': bilan_stats["building_breakdown"]["by_building_type"][b_type]["conso_2024"],
                'Consommation 2050 (MWh)': bilan_stats["building_breakdown"]["by_building_type"][b_type]["conso_2050"],
                'Part 2024 (%)': bilan_stats["building_breakdown"]["by_building_type"][b_type]["share_2024"],
                'Part 2050 (%)': bilan_stats["building_breakdown"]["by_building_type"][b_type]["share_2050"]
            })
        
        bld_df = pd.DataFrame(bld_data).set_index('Type')
        
        # Show metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Consommation par type**")
            fig = px.pie(bld_df, values='Part 2024 (%)', names=bld_df.index,
                        title="Répartition initiale (2024)")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Consommation par type**")
            fig = px.pie(bld_df, values='Part 2050 (%)', names=bld_df.index,
                        title="Répartition final (2050)")
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            st.markdown("**Évolution des consommations**")
            fig = px.bar(bld_df[['Consommation 2024 (MWh)', 'Consommation 2050 (MWh)']],
                        barmode='group', labels={'value': 'MWh'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Show detailed table
        bld_df['Réduction (MWh)'] = bld_df['Consommation 2024 (MWh)'] - bld_df['Consommation 2050 (MWh)']
        bld_df['Réduction (%)'] = (bld_df['Réduction (MWh)'] / bld_df['Consommation 2024 (MWh)']) * 100
        st.dataframe(bld_df.style.format("{:,.0f}"), use_container_width=True)
    
    # === SECTION 5: ADDITIONAL INSIGHTS ===
    st.markdown("### 🔍 Analyses complémentaires")
    
    # Energy transition indicators
    with st.expander("Indicateurs de transition énergétique"):
        st.markdown("""
        - **Taux de décarbonation**: {:.1f}% de réduction des émissions
        - **Diversification énergétique**: Evolution du mix énergétique
        - **Efficacité énergétique**: {:.1f}% d'amélioration de l'intensité carbone
        """.format(
            bilan_stats['Réduction émissions (%)'],
            bilan_stats['Réduction intensité (%)']
        ))
    
    # Performance comparison
    with st.expander("Comparaison sectorielle"):
        if bilan_stats["building_breakdown"]:
            residential_red = (bld_df.loc['Résidentiel', 'Réduction (MWh)'] / 
                             bld_df.loc['Résidentiel', 'Consommation 2024 (MWh)']) * 100
            tertiary_red = (bld_df.loc['Tertiaire', 'Réduction (MWh)'] / 
                          bld_df.loc['Tertiaire', 'Consommation 2024 (MWh)']) * 100
            
            st.markdown(f"""
            - **Secteur résidentiel**: {residential_red:.1f}% de réduction
            - **Secteur tertiaire**: {tertiary_red:.1f}% de réduction
            """)
        else:
            st.info("Données par type de bâtiment non disponibles")

def display_detailed_tables(conso_par_vecteur, emissions_par_vecteur, annees):
    """Affiche les tableaux détaillés par énergie"""
    st.subheader("Détails par type d'énergie")
    
    tab1, tab2 = st.tabs(["Consommations", "Émissions"])
    
    # Tableau des consommations
    with tab1:
        conso_df = pd.DataFrame(conso_par_vecteur, index=annees.astype(str))
        conso_df.index.name = "Année"
        st.dataframe(conso_df.style.format("{:,.1f}"), use_container_width=True)
    
    # Tableau des émissions
    with tab2:
        emissions_df = pd.DataFrame(emissions_par_vecteur, index=annees.astype(str))
        emissions_df.index.name = "Année"
        st.dataframe(emissions_df.style.format("{:,.1f}"), use_container_width=True)

def display_assumptions(heating_efficiency_map, electricity_carbone_factor, 
                       facteurs_carbone, annees):
    """Affiche la section des hypothèses et paramètres utilisés"""
    st.header("🔎 Hypothèses")
    
    # === ÉVOLUTION DES FACTEURS CARBONE ===
    st.subheader("Évolution des facteurs carbone (kgCO₂/kWh)")
    
    carbone_df = pd.DataFrame({
        "Électricité": electricity_carbone_factor
    }, index=annees)
    carbone_df.index.name = "Année"
    
    fig_carbone = px.line(
        carbone_df,
        title="Évolution du facteur carbone pour l'électricité<br><sup>Rapport HySPI Hydrogène industriel - Scénarios prospectifs des impacts environnementaux</sup>",
        labels={"value": "kgCO₂/kWh", "variable": "Énergie"}
    )
    fig_carbone.update_layout(height=500)
    st.plotly_chart(fig_carbone, use_container_width=True)

    # === TABLEAU DES RENDEMENTS ===
    st.subheader("Tableau résumé des hypothèses de rendement et d'émissions carbone de chauffage")
    df_eff = pd.DataFrame({
        "Énergie": list(heating_efficiency_map.keys()),
        "Rendement (%)": [f"{eff:.2f}" for eff in heating_efficiency_map.values()],
        "Émissions (kgCO₂/kWh)": [
            facteurs_carbone.get(energie, [0])[0] 
            for energie in heating_efficiency_map.keys()
        ]
    })
    st.table(df_eff)

def save_session_data(scenario_name, strategy_name, usage_selection, 
                     substitutions, coverage_rates):
    """Sauvegarde les données de session pour utilisation dans d'autres pages"""
    st.session_state['scenario_name'] = scenario_name
    st.session_state['strategy_name'] = strategy_name
    st.session_state['usage_selection'] = usage_selection
    st.session_state['substitutions'] = substitutions
    st.session_state['coverage_rates'] = coverage_rates

# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def main():
    """Fonction principale de l'application"""
    
    # === CONFIGURATION DE LA PAGE ===
    setup_page_header()
    display_introduction()
    st.markdown("---")  # Séparateur visuel
    
    # === INITIALISATION DES DONNÉES ===
    original_city_data = initialize_data()
    
    # === CONFIGURATION DU PÉRIMÈTRE D'ÉTUDE ===
    usage_selection, cud_only = setup_study_perimeter_selection()
    
    # Application des filtres sur les données
    st.session_state.city_data = filter_data_by_selection(
        original_city_data, usage_selection, cud_only
    )
    
    # Vérification que des données existent après filtrage
    if len(st.session_state.city_data) == 0:
        st.error("❌ Aucun bâtiment ne correspond aux critères sélectionnés. Veuillez modifier vos filtres.")
        st.stop()
    
    # Affichage des statistiques des données filtrées
    display_filtered_data_stats(st.session_state.city_data, usage_selection, cud_only)
    
    # === CALCULS PRÉPARATOIRES ===
    heating_efficiency_map = calculate_heating_efficiencies(st.session_state.city_data)
    strategies = prepare_strategies(st.session_state.city_data)
    vecteurs_energie = st.session_state.city_data["energie_imope"].unique()
    original_profile = calculate_energy_profile_by_sector(st.session_state.city_data)
    
    # === PARAMÉTRAGE DE LA SIMULATION ===
    coverage_rates  = setup_simulation_parameters()
    selected_strategy = setup_renovation_strategy_selection(strategies)
    selected_scenario = setup_temporal_scenario_selection()
    substitutions = setup_energy_substitution(original_profile, heating_efficiency_map)

    # === INITIALISATION DE L'ÉTAT DE SESSION ===
    if "simulation_already_run" not in st.session_state:
        st.session_state.simulation_already_run = False

    # === BOUTON DE LANCEMENT ===
    run_simulation = False

    # Simulation automatique au premier chargement
    if not st.session_state.simulation_already_run:
        run_simulation = True
        st.session_state.simulation_already_run = True

    # Simulation manuelle via bouton
    st.sidebar.markdown("---")
    if st.sidebar.button("🚀 Lancer ou relancer la simulation"):
        run_simulation = True
        st.session_state.simulation_already_run = True  # Important pour afficher ensuite les résultats

    if run_simulation:
        strategie = strategies[selected_strategy]
        scenario_temporelles = scenarios_temporelles[selected_scenario]

        df_simulation, conso_par_vecteur, emissions_par_vecteur = simulate(
            strategie,
            coverage_rates,
            scenario_temporelles,
            vecteurs_energie,
            heating_efficiency_map,
            substitutions.copy()
        )
        save_session_data(
            selected_scenario, selected_strategy, usage_selection, 
            substitutions, coverage_rates
        )
        
        # Stocker résultats dans session pour réutilisation
        st.session_state.conso_par_vecteur = conso_par_vecteur
        st.session_state.emissions_par_vecteur = emissions_par_vecteur
        st.session_state.strategie = strategie
        st.session_state.scenario_temporelles = scenario_temporelles
        st.session_state.city_data = df_simulation

    # Si simulation pas lancée maintenant, vérifier si résultats en session
    if not run_simulation:
        if (
            "conso_par_vecteur" in st.session_state 
            and "emissions_par_vecteur" in st.session_state
            and "strategie" in st.session_state
            and "scenario_temporelles" in st.session_state
        ):
            conso_par_vecteur = st.session_state.conso_par_vecteur
            emissions_par_vecteur = st.session_state.emissions_par_vecteur
            strategie = st.session_state.strategie
            scenario_temporelles = st.session_state.scenario_temporelles
            city_data = st.session_state.city_data
        else:
            st.info("Cliquez sur le bouton 🚀 dans la sidebar pour lancer la simulation.")
            st.stop()

    # Maintenant que les variables sont définies, on peut afficher
    display_results(
        conso_par_vecteur, 
        emissions_par_vecteur, strategie, annees, scenario_temporelles
    )
    st.markdown("---") 
    display_summary_metrics(df_simulation, conso_par_vecteur, emissions_par_vecteur)
    st.markdown("---") 
    display_detailed_tables(conso_par_vecteur, emissions_par_vecteur, annees)
    st.markdown("---")  # Séparateur visuel
    # Affichage des hypothèses et paramètres
    display_assumptions(
        heating_efficiency_map, 
        electricity_carbone_factor, 
        facteurs_carbone, 
        annees
    )


# ============================================================================
# POINT D'ENTRÉE DE L'APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()