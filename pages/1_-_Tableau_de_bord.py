"""
# SmartDash - Simulation de rénovation énergétique et émissions carbone
# This application simulates the impact of energy renovation strategies on energy consumption and carbon emissions in a city.
# It allows users to choose different renovation strategies and scenarios, and visualizes the results.  
To run this application, ensure you have the required libraries installed:
# pip install streamlit 
Use the following command to run the application:
# streamlit run SmartDash.py 

"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import warnings

warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")

# Configuration de la page
st.set_page_config(
    page_title="SmartE - Simulation Rénovation Énergétique",
    page_icon="images/logo.png",
    layout="wide"
)

# =========================
# Paramètres globaux
# =========================

annees = np.arange(2024, 2051)
n_annees = len(annees)
coverage_rate = 1  # Taux de rénovation maximum
substitution_max_rate = 0.5  # Taux maximum de substitution gaz -> électricité

# Scénarios de rénovation
scenarios = {
    "Linéaire": np.linspace(0, coverage_rate, n_annees),
    "Rapide au début": np.array([coverage_rate * (1 - np.exp(-0.15 * i)) for i in range(n_annees)]),
    "Lent au début": np.array([coverage_rate * (1 - np.exp(-0.05 * i)) for i in range(n_annees)]),
}

# Facteurs carbone dynamiques (kgCO₂/kWh)
electricity_carbone_factor = [
    0.045511605,
    0.043359175, 0.041206744, 0.039054314, 0.036901883, 0.034749453,
    0.032597022, 0.030700969, 0.028804917, 0.026908864, 0.025012811,
    0.023116759, 0.021220706, 0.019324653, 0.0174286, 0.015532548,
    0.013636495, 0.013559714, 0.013482933, 0.013406151, 0.01332937,
    0.013252589, 0.013175808, 0.013099027, 0.013022245, 0.012945464,
    0.012868683
]

facteurs_carbone = {
    "électricité": electricity_carbone_factor,
    "gaz naturel": np.full(n_annees, 0.23),
    "fioul": np.full(n_annees, 0.300),
    "bois": np.full(n_annees, 0.025),
    "chauffage urbain": np.linspace(0.3, 0.2, n_annees),
    "autre": np.full(n_annees, 0.150),
    "mixte": np.full(n_annees, 0.100),
    "PAC Air-Air": electricity_carbone_factor,
    "PAC Air-Eau": electricity_carbone_factor,
    "PAC Eau-Eau": electricity_carbone_factor,
    "PAC Géothermique": electricity_carbone_factor,
}

# =========================
# Fonctions
# =========================

def substitute_energy(df, source_vec, target_vec, year_index, total_years, max_substitution_rate, efficiency_map):
    """
    Substitution progressive d'une énergie source vers une énergie cible en tenant compte des rendements.
    """
    substitution_ratio = max_substitution_rate * (year_index / (total_years - 1))
    source_energy = df[source_vec]
    energy_to_substitute = source_energy * substitution_ratio
    
    # Correction des rendements
    source_eff = efficiency_map.get(source_vec, 1.0)
    target_eff = efficiency_map.get(target_vec, 1.0)
    substituted_energy = energy_to_substitute * (source_eff / target_eff)
    
    # Mise à jour des consommations
    df[source_vec] -= energy_to_substitute
    if target_vec not in df:
        df[target_vec] = 0.0
    df[target_vec] += substituted_energy
    
    return df

def calculate_heating_efficiencies(df):
    """Calcule les rendements moyens de chauffage par type d'énergie."""
    efficiencies = {}
    for vec in df["energie_imope"].unique():
        mean_efficiency = df[df["energie_imope"] == vec]["heating_efficiency"].mean()
        efficiencies[vec] = mean_efficiency
    return efficiencies

def prepare_strategies(df):
    """Prépare les différentes stratégies de rénovation."""
    df_tri_m2 = df.copy()
    df_tri_m2["conso_m2"] = df_tri_m2["Consommation par m² par an (en kWh/m².an)_basic"]
    df_tri_m2 = df_tri_m2.sort_values(by="conso_m2", ascending=False).reset_index(drop=True)

    df_tri_MWh = df.sort_values(by="total_energy_consumption_basic", ascending=False).reset_index(drop=True)
    df_random = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return {
        "Tri conso spécifique (kWh/m².an)": df_tri_m2,
        "Tri conso annuelle (MWh/an)": df_tri_MWh,
        "Ordre aléatoire": df_random,
    }

def calculate_energy_profile_by_sector(df, sector_column="energie_imope"):
    """Calcule le profil énergétique par secteur (type d'énergie)"""
    profile = {}
    
    for energie in df[sector_column].unique():
        mask = df[sector_column] == energie
        profile[energie] = {
            "consommation_basic": df[mask]["total_energy_consumption_basic"].sum() / 1000,  # MWh
            "consommation_renovated": df[mask]["total_energy_consumption_renovated"].sum() / 1000,  # MWh
            "nb_batiments": len(df[mask]),
            "consommation_moyenne": df[mask]["total_energy_consumption_basic"].mean(),
            "potentiel_economie": (df[mask]["total_energy_consumption_basic"].sum() - 
                                 df[mask]["total_energy_consumption_renovated"].sum()) / 1000
        }
    
    return profile

def simulate(df, scenario, vecteurs_energie, efficiency_map, substitutions):
    """Exécute la simulation pour une stratégie et un scénario donnés avec substitutions dynamiques."""
    conso_par_vecteur = {vec: [] for vec in vecteurs_energie}
    emissions_par_vecteur = {vec: [] for vec in vecteurs_energie}
    n_batiments = len(df)

    for i_annee, p in enumerate(scenario):
        n_reno = int(p * n_batiments)
        df_reno = df.iloc[:n_reno]
        df_non_reno = df.iloc[n_reno:]

        conso_vect = {vec: 0.0 for vec in vecteurs_energie}

        for vec in vecteurs_energie:
            conso_reno = df_reno[df_reno["energie_imope"] == vec]["total_energy_consumption_renovated"].sum() / 1000
            conso_rest = df_non_reno[df_non_reno["energie_imope"] == vec]["total_energy_consumption_basic"].sum() / 1000
            conso_vect[vec] = conso_reno + conso_rest

        # Substitution gaz -> électricité
        for source_vec, target_vec, percentage in substitutions:
            conso_vect = substitute_energy(
                conso_vect,
                source_vec=source_vec,
                target_vec=target_vec,
                year_index=i_annee,
                total_years=n_annees,
                max_substitution_rate=percentage / 100,  # conversion en fraction
                efficiency_map=efficiency_map
            )

        # Stockage des consommations et émissions
        for vec in vecteurs_energie:
            conso_par_vecteur[vec].append(conso_vect[vec])
            facteur = facteurs_carbone.get(vec, np.zeros(n_annees))[i_annee]
            emissions = conso_vect[vec] * facteur  # tCO₂
            emissions_par_vecteur[vec].append(emissions)

    return conso_par_vecteur, emissions_par_vecteur

def synthesize_results(conso_par_vecteur, emissions_par_vecteur):
    """Calcule les bilans énergétiques et carbone."""
    total_conso_2024 = sum([conso[0] for conso in conso_par_vecteur.values()])
    total_conso_2050 = sum([conso[-1] for conso in conso_par_vecteur.values()])
    total_emi_2024 = sum([em[0] for em in emissions_par_vecteur.values()])
    total_emi_2050 = sum([em[-1] for em in emissions_par_vecteur.values()])

    return {
        "Consommation 2024 (MWh)": total_conso_2024,
        "Consommation 2050 (MWh)": total_conso_2050,
        "Réduction conso (MWh)": total_conso_2024 - total_conso_2050,
        "Réduction conso (%)": 100 * (total_conso_2024 - total_conso_2050) / total_conso_2024,
        "Émissions 2024 (tCO₂)": total_emi_2024,
        "Émissions 2050 (tCO₂)": total_emi_2050,
        "Réduction émissions (%)": 100 * (total_emi_2024 - total_emi_2050) / total_emi_2024,
    }

def create_consumption_chart(annees, conso_par_vecteur, title="Consommation par énergie"):
    """Crée un graphique de consommation par énergie"""
    fig = go.Figure()
    
    for vec, consos in conso_par_vecteur.items():
        fig.add_trace(go.Scatter(
            x=annees,
            y=consos,
            name=vec,
            stackgroup='one',
            line_shape='spline'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Année",
        yaxis_title="Consommation (MWh)",
        hovermode="x unified",
        height=500
    )
    return fig

def create_emissions_chart(annees, emissions_par_vecteur, title="Émissions carbone par énergie"):
    """Crée un graphique d'émissions par énergie"""
    fig = go.Figure()
    
    for vec, emissions in emissions_par_vecteur.items():
        fig.add_trace(go.Scatter(
            x=annees,
            y=emissions,
            name=vec,
            stackgroup='one',
            line_shape='spline'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Année",
        yaxis_title="Émissions (tCO₂)",
        hovermode="x unified",
        height=500
    )
    return fig

@st.cache_data
def load_sample_data():
    """Charge et concatène les trois parties du fichier city_file."""
    part1 = pd.read_pickle("city_part1.pkl")
    part2 = pd.read_pickle("city_part2.pkl")
    part3 = pd.read_pickle("city_part3.pkl")
    #"PAC Air-Air" : 3, "PAC Air-Eau" : 3, "PAC Eau-Eau" : 4, "PAC Géothermique" : 5
    part4 = pd.DataFrame({
        "total_energy_consumption_basic": [0, 0, 0, 0],
        "Consommation par m² par an (en kWh/m².an)_basic": [0, 0, 0, 0],
        "total_energy_consumption_renovated": [0, 0, 0, 0],
        "energie_imope": ["PAC Air-Air", "PAC Air-Eau", "PAC Eau-Eau", "PAC Géothermique"],
        "heating_efficiency": [3.0, 3.0, 4.0, 5.0]
    })  # Ajout d'une ligne vide pour éviter les erreurs de concaténation

    city = pd.concat([part1, part2, part3,part4], ignore_index=True)
    return city

# =========================
# Application Streamlit
# =========================

def main():
    # En-tête avec colonnes pour logo + titre
    col_logo, col_title = st.columns([1, 5])
    with col_logo:
        st.image("images/logo.png", width=140)
    with col_title:
        st.title("SmartE - Simulateur Rénovation Énergétique")
    
    # Bannière d'introduction
    st.markdown("""
    <div style="background-color:#f0f2f6;padding:20px;border-radius:10px;margin-bottom:20px">
    <h3 style="color:#1e3a8a;margin-top:0;"> Optimisez votre stratégie de rénovation énergétique</h3>
    <p>Cet outil permet de simuler différents scénarios de rénovation du parc immobilier et d'analyser leur impact sur :</p>
    <ul>
        <li>Les <b>consommations énergétiques</b> (MWh)</li>
        <li>Les <b>émissions de CO₂</b> (tonnes équivalent carbone)</li>
        <li>La <b>transition énergétique</b> du territoire</li>
    </ul>
    <p>Période d'analyse : <b>2024 à 2050</b> (objectif neutralité carbone)</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Section "Comment ça marche"
    with st.expander("ℹ️ Comment utiliser ce simulateur", expanded=False):
        st.markdown("""
        1. **Paramétrez** votre simulation dans la barre latérale ←
        2. **Visualisez** les résultats en temps réel
        3. **Comparez** différents scénarios
        
        ### Concepts clés :
        - **Stratégies de rénovation** : Ordre de priorité des bâtiments à rénover  
        - **Scénarios temporels** : Rythme de déploiement des rénovations  
        - **Conversion énergétique** : Substitution entre sources d'énergie  
        - **Facteurs carbone** : Intensité CO₂ de chaque énergie (kgCO₂/kWh)
        """)
    
    # Séparateur visuel
    st.markdown("---")

    
    # Chargement des données
    if 'city_data' not in st.session_state:
        with st.spinner("Chargement des données..."):
            st.session_state.city_data = load_sample_data()
    
    city_data = st.session_state.city_data
    
    # Calcul des rendements moyens par type d'énergie
    heating_efficiency_map = calculate_heating_efficiencies(city_data)
    
    # Préparer les stratégies
    strategies = prepare_strategies(city_data)
    vecteurs_energie = city_data["energie_imope"].unique()

    # Calcul du profil énergétique initial
    original_profile = calculate_energy_profile_by_sector(city_data)
    
    # Sidebar pour les paramètres
    st.sidebar.header("🔧 Paramètres de modification")
    
    # Informations générales
    st.sidebar.markdown("### 📊 Données de base")
    st.sidebar.metric("Nombre de bâtiments", len(city_data))
    st.sidebar.metric("Consommation totale", f"{sum([p['consommation_basic'] for p in original_profile.values()]):.0f} MWh/an")
    

    st.sidebar.markdown("### 📅 Stratégie de rénovation")


    # Taux de rénovation avec explication
    with st.sidebar.expander("🏗️ Taux de rénovation", expanded=False):
        st.markdown("""
        **Définit la proportion totale du parc immobilier qui sera rénovée d'ici 2050.**
        - 0% = Aucune rénovation
        - 100% = Tous les bâtiments rénovés
        """)
        coverage_rate = st.slider(
            "Taux de rénovation total (2024-2050)",
            min_value=0,
            max_value=100,
            value=30,
            step=5,
            help="Pourcentage total du parc immobilier à rénover sur la période 2024-2050"
        )
        
        # Mise à jour des scénarios avec le taux de rénovation
        for key in scenarios:
            scenarios[key] = scenarios[key] * (coverage_rate / 100.0)
    
    # Choix de la stratégie avec explication
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
    
    # Choix du scénario avec explication
    with st.sidebar.expander("⏱️ Scénario temporel", expanded=False):
        st.markdown("""
        **Définit la vitesse de déploiement des rénovations:**
        - 🏁 Linéaire: Progression constante
        - 🚀 Rapide début: Effort important dès le début
        - 🐢 Lent début: Démarrage progressif
        """)
        selected_scenario = st.selectbox(
            "Profil de déploiement",
            list(scenarios.keys()),
            index=0,
            help="Rythme de déploiement des rénovations sur la période"
        )

    # Système de substitution des systèmes d'énergie
    # Section Substitution d'énergies
    with st.sidebar.expander("🔄 Conversion des énergies de chauffage", expanded=False):

        substitutions = []

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

        # Conteneur pour la substitution
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
        # Bouton d'ajout
        if st.button("➕ Ajouter cette conversion", 
                    use_container_width=True,
                    help="Valider cette règle de conversion"):
            if rate > 0:
                substitutions.append((source, target, rate))
                st.success(f"Conversion ajoutée : {rate}% de {source} → {target}")
            else:
                st.warning("Veuillez spécifier un pourcentage de conversion")

        # Affichage des conversions actives
        if substitutions:
            st.markdown("**Conversions actives :**")
            for i, (src, tgt, pct) in enumerate(substitutions):
                cols = st.columns([4, 1])
                with cols[0]:
                    st.markdown(f"`{src}` → `{tgt}` ({pct}%)")
                with cols[1]:
                    if st.button("✕", key=f"del_{i}"):
                        substitutions.pop(i)
                        st.rerun()
    
    # Simulation
    df_selected = strategies[selected_strategy]
    scenario_selected = scenarios[selected_scenario]
    
    conso_par_vecteur, emissions_par_vecteur = simulate(
        df_selected,
        scenario_selected,
        vecteurs_energie,
        heating_efficiency_map,
        substitutions
    )

    # Affichage des résultats
    st.subheader(f"Résultats pour la stratégie: {selected_strategy} - Scénario: {selected_scenario}")
    
    # Graphiques de consommation et émissions
    col1, col2 = st.columns(2)
    
    with col1:
        fig_conso = create_consumption_chart(annees, conso_par_vecteur)
        st.plotly_chart(fig_conso, use_container_width=True)
    
    with col2:
        fig_emissions = create_emissions_chart(annees, emissions_par_vecteur)
        st.plotly_chart(fig_emissions, use_container_width=True)
    
    # Bilan énergétique et carbone
    st.subheader("Bilan énergétique et carbone (2024-2050)")
    bilan_stats = synthesize_results(conso_par_vecteur, emissions_par_vecteur)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Consommation 2024",
            f"{bilan_stats['Consommation 2024 (MWh)']:,.0f} MWh"
        )
        st.metric(
            "Consommation 2050",
            f"{bilan_stats['Consommation 2050 (MWh)']:,.0f} MWh",
            delta=f"-{bilan_stats['Réduction conso (MWh)']:,.0f} MWh ({bilan_stats['Réduction conso (%)']:.1f}%)"
        )
    
    with col2:
        st.metric(
            "Émissions 2024",
            f"{bilan_stats['Émissions 2024 (tCO₂)']:,.0f} tCO₂"
        )
        st.metric(
            "Émissions 2050",
            f"{bilan_stats['Émissions 2050 (tCO₂)']:,.0f} tCO₂",
            delta=f"-{bilan_stats['Émissions 2024 (tCO₂)'] - bilan_stats['Émissions 2050 (tCO₂)']:,.0f} tCO₂ ({bilan_stats['Réduction émissions (%)']:.1f}%)"
        )
    
    with col3:
        total_conso = sum([sum(conso) for conso in conso_par_vecteur.values()])
        total_emissions = sum([sum(emissions) for emissions in emissions_par_vecteur.values()])
        st.metric("Consommation totale (2024-2050)", f"{total_conso:,.0f} MWh")
        st.metric("Émissions totales (2024-2050)", f"{total_emissions:,.0f} tCO₂")
    
    # Détails par énergie
    st.subheader("Détails par type d'énergie")
    
    tab1, tab2 = st.tabs(["Consommations", "Émissions"])
    
    with tab1:
        conso_df = pd.DataFrame(conso_par_vecteur, index=annees)
        conso_df.index.name = "Année"
        st.dataframe(conso_df.style.format("{:,.1f}"), use_container_width=True)
    
    with tab2:
        emissions_df = pd.DataFrame(emissions_par_vecteur, index=annees)
        emissions_df.index.name = "Année"
        st.dataframe(emissions_df.style.format("{:,.1f}"), use_container_width=True)
    
    st.header("📊 Hypothèses")
    # Facteurs carbone  Rapport HySPI Hydrogène industriel - Scénarios prospectifs des impacts environnementaux
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

    # Affichage des rendements
    st.subheader("Rendements moyens")
    df_eff = pd.DataFrame(
        {"Energie": list(heating_efficiency_map.keys()),
         "Rendement": [f"{eff:.2f}" for eff in heating_efficiency_map.values()]
        }
    )
    st.table(df_eff)

if __name__ == "__main__":
    main()