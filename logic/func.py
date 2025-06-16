import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from logic.param import annees, facteurs_carbone, n_annees

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

def simulate(df, coverage, scenario_temporel, vecteurs_energie, efficiency_map, substitutions):
    """Exécute la simulation pour une stratégie et un scénario donnés avec substitutions dynamiques."""
    conso_par_vecteur = {vec: [] for vec in vecteurs_energie}
    emissions_par_vecteur = {vec: [] for vec in vecteurs_energie}
    n_batiments = len(df)
    n_annees = len(scenario_temporel)

    # Normalisation du scénario temporel selon le taux de couverture
    scenario_temporel = scenario_temporel * (coverage / 100.0)

    for i_annee, p in enumerate(scenario_temporel):
        #print(i_annee, p)
        n_reno = int(p * n_batiments)
        df_reno = df.iloc[:n_reno]
        df_non_reno = df.iloc[n_reno:]

        conso_vect = {vec: 0.0 for vec in vecteurs_energie}

        # Calcul des consommations de base (sans substitution)
        for vec in vecteurs_energie:
            conso_reno = df_reno[df_reno["energie_imope"] == vec]["total_energy_consumption_renovated"].sum() / 1000
            conso_rest = df_non_reno[df_non_reno["energie_imope"] == vec]["total_energy_consumption_basic"].sum() / 1000
            conso_vect[vec] = conso_reno + conso_rest
            #print(f"Année {annees[i_annee]} - {vec} : {conso_reno} + {conso_rest} = {conso_vect[vec]} MWh")


        # Application des substitutions avec progression selon le scénario temporel
        for source_vec, target_vec, percentage in substitutions:
            # Calcul du taux effectif de substitution pour cette année
            # en fonction de la progression du scénario temporel
            progression_scenario = scenario_temporel[i_annee] / scenario_temporel[-1]  # Progression relative (0 à 1)
            effective_substitution_rate = (percentage / 100) * progression_scenario
            
            conso_vect = substitute_energy(
                conso_vect,
                source_vec=source_vec,
                target_vec=target_vec,
                year_index=i_annee,
                total_years=n_annees,
                max_substitution_rate=effective_substitution_rate,  # Taux ajusté par le scénario
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

def create_cumulative_emissions_chart(annees, emissions_par_vecteur, scenario_name):
    """Crée un graphique d'émissions cumulées avec effet de la rénovation"""
    
    # Calcul des émissions cumulées par énergie
    cumulative_emissions = {
        energie: np.cumsum(emissions) 
        for energie, emissions in emissions_par_vecteur.items()
    }
    
    # Création du graphique empilé
    fig = go.Figure()
    
    # Ajout de chaque énergie comme une couche empilée
    for energie in emissions_par_vecteur.keys():
        fig.add_trace(go.Scatter(
            x=annees,
            y=cumulative_emissions[energie],
            name=energie,
            stackgroup='one',
            line=dict(width=0.5),
            mode='lines'
        ))
    
    # Mise en forme du graphique
    fig.update_layout(
        title=f"Émissions cumulées de CO₂ (2024-2050) - Scénario {scenario_name}",
        xaxis_title="Année",
        yaxis_title="Émissions cumulées (tonnes CO₂)",
        hovermode="x unified",
        legend_title="Type d'énergie",
        height=500,
        annotations=[
            dict(
                text="Plus la courbe est plate, plus les réductions d'émissions sont importantes",
                xref="paper", yref="paper",
                x=0.5, y=1.1,
                showarrow=False,
                font=dict(size=10)
            )
        ]
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
        "total_energy_consumption_basic": [0, 0, 0,0,0,0],
        "Consommation par m² par an (en kWh/m².an)_basic": [0, 0, 0,0, 0, 0],
        "total_energy_consumption_renovated": [0, 0, 0,0, 0, 0],
        "Consommation par m² par an (en kWh/m².an)": [0, 0, 0,0, 0, 0],
        "energie_imope": ["PAC Air-Air", "PAC Air-Eau", "PAC Géothermique","PAC Air-Air", "PAC Air-Eau", "PAC Géothermique"],
        "heating_efficiency": [3.0, 4.0, 5.0,3.0, 4.0, 5.0],
        "UseType": ["LOGEMENT", "LOGEMENT", "LOGEMENT","Autre", "Autre", "Autre"],
    })  # Ajout d'une ligne vide pour éviter les erreurs de concaténation

    city = pd.concat([part1, part2, part3,part4], ignore_index=True)
    return city


def filter_data_by_selection(city_data: pd.DataFrame, usage_selection: str, cud_only: bool = False):
    """
    Filtre les données selon la sélection d'usage et optionnellement sur les bâtiments CUD.
    
    Args:
        city_data: DataFrame contenant toutes les données
        usage_selection: "Résidentiel uniquement" ou "Résidentiel + Tertiaire"
        cud_only: Si True, ne garde que les bâtiments CUD
    
    Returns:
        pd.DataFrame: Données filtrées
    """
    
    # Filtre CUD si demandé (supposant qu'il existe une colonne 'is_cud' ou similaire)
    if cud_only:
        if 'is_cud' in city_data.columns:
            filtered_data = city_data[city_data['is_cud'] == True].copy()
        else:
            # Si la colonne n'existe pas, on peut utiliser d'autres critères
            # Par exemple, si les bâtiments CUD ont un identifiant spécifique
            st.warning("⚠️ En développement : filtre CUD non disponible. Toutes les données seront utilisées.")
            filtered_data = city_data.copy()
    else:
        filtered_data = city_data.copy()
    
    # Filtre par type d'usage
    if usage_selection == "Résidentiel":
        filtered_data = filtered_data[filtered_data["UseType"] == "LOGEMENT"]
    elif usage_selection == "Tertiaire":
        filtered_data = filtered_data[filtered_data["UseType"] != "LOGEMENT"]
    elif usage_selection == "Résidentiel + Tertiaire":
        pass
    return filtered_data
    
def get_building_consumption_distribution(df, scenario, year_index):
    """
    Calcule la distribution des consommations par m² pour une année donnée de la simulation.
    
    Args:
        df: DataFrame des bâtiments triés selon la stratégie
        scenario: Scénario de rénovation (array des pourcentages)
        year_index: Index de l'année dans le scénario
    
    Returns:
        dict: Distribution des consommations par m² avec statut de rénovation
    """
    n_batiments = len(df)
    p = scenario[year_index] if year_index < len(scenario) else scenario[-1]
    n_reno = int(p * n_batiments)
    
    # Séparer les bâtiments rénovés et non rénovés
    df_reno = df.iloc[:n_reno]
    df_non_reno = df.iloc[n_reno:]

    # Créer un DataFrame temporaire avec les consommations
    # Gérer les NaNs potentiels dans les colonnes sources avant concaténation
    reno_consumption = df_reno['Consommation par m² par an (en kWh/m².an)'].fillna(0)
    non_reno_consumption = df_non_reno['Consommation par m² par an (en kWh/m².an)_basic'].fillna(0)

    df_temp = pd.DataFrame({
        'renovated': [True] * len(df_reno) + [False] * len(df_non_reno),
        'energy_type': pd.concat([df_reno['energie_imope'], df_non_reno['energie_imope']]),
        'consumption_m2': pd.concat([reno_consumption, non_reno_consumption])
    })

    # Clipper les valeurs pour correspondre au range_x de l'histogramme et s'assurer que tous les bâtiments sont comptés.
    # Le range_x dans create_dynamic_histogram est [0, 600]
    df_temp['consumption_m2'] = df_temp['consumption_m2'].clip(lower=0, upper=600)

    return {
        'consumption_m2': df_temp['consumption_m2'].values,
        'renovated': df_temp['renovated'].values,
        'energy_type': df_temp['energy_type'].values,
        'n_renovated': n_reno,
        'n_total': n_batiments
    }

def create_evolution_chart(annees, df, scenario, title="Évolution des consommations"):
    """
    Préparer les données pour toutes les années
    """
    evolution_data = []
    for i, year in enumerate(annees):
        dist_data = get_building_consumption_distribution(df, scenario, i)
        year_df = pd.DataFrame({
            'renovated': dist_data['renovated'],
            'energy_type': dist_data['energy_type'],
            'year': year
        })
        energy_counts = year_df.groupby(['energy_type', 'renovated']).size().reset_index(name='count')
        energy_counts['year'] = year
        evolution_data.append(energy_counts)
    
    # Combiner toutes les années
    all_evolution_df = pd.concat(evolution_data, ignore_index=True)
    
    # Créer le statut combiné pour la couleur
    all_evolution_df['status_energy'] = all_evolution_df['energy_type'] + ' - ' + \
                                       all_evolution_df['renovated'].map({True: 'Rénovés', False: 'Non rénovés'})
    
    # Créer le graphique
    fig = px.area(
        all_evolution_df,
        x='year',
        y='count',
        color='status_energy',
        title=title,
        labels={
            'year': 'Année',
            'count': 'Nombre de bâtiments',
            'status_energy': 'Type d\'énergie - Statut'
        }
    )
    
    # Personnaliser le style
    fig.update_layout(
        height=400,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.05
        ),
        showlegend=True
    )
    
    return fig

def create_dynamic_histogram(df, scenario_temporel, title="Distribution des consommations par m²"):
    """
    Crée un histogramme animé montrant l'évolution de la distribution des consommations par m².
    Cette version utilise un binning manuel et px.bar pour garantir une largeur de barre constante.
    """
    scenario_temporel = scenario_temporel * (st.session_state["coverage_rate"] / 100.0)
    # 1. Obtenir les données pour toutes les années
    all_data = []
    for i, year in enumerate(annees):
        dist_data = get_building_consumption_distribution(df, scenario_temporel, i)
        year_df = pd.DataFrame({
            'consumption_m2': dist_data['consumption_m2'],
            'renovated': dist_data['renovated'],
            'year': year
        })
        all_data.append(year_df)
    full_df = pd.concat(all_data, ignore_index=True)

    # 2. Définir les classes (bins) manuellement
    bin_size = 10  # Taille de chaque classe
    max_val = 800
    bins = np.arange(0, max_val + bin_size, bin_size)
    bin_labels = [f'{i}-{i+bin_size}' for i in bins[:-1]]

    # 3. Assigner chaque bâtiment à une classe
    full_df['bin'] = pd.cut(full_df['consumption_m2'], bins=bins, labels=bin_labels, right=False, include_lowest=True)

    # 4. Compter les bâtiments par classe, année et statut
    binned_counts = full_df.groupby(['year', 'bin', 'renovated'], observed=True).size().reset_index(name='count')

    # 5. Assurer que chaque classe existe pour chaque année (même avec un compte de 0)
    all_years = full_df['year'].unique()
    all_statuses = [True, False]
    complete_grid = pd.MultiIndex.from_product(
        [all_years, bin_labels, all_statuses],
        names=['year', 'bin', 'renovated']
    ).to_frame(index=False)

    final_df = pd.merge(complete_grid, binned_counts, on=['year', 'bin', 'renovated'], how='left').fillna(0)
    
    final_df['bin'] = pd.Categorical(final_df['bin'], categories=bin_labels, ordered=True)
    final_df.sort_values(by=['year', 'bin'], inplace=True)

    # 6. Créer le graphique à barres animé avec px.bar
    final_df['renovated_label'] = final_df['renovated'].map({True: 'Rénové', False: 'Non-rénové'})

    fig = px.bar(
        final_df,
        x='bin',
        y='count',
        color='renovated_label',
        animation_frame='year',
        title=title,
        labels={
            'bin': 'Consommation par m² (kWh/m².an)',
            'count': 'Nombre de bâtiments',
            'renovated_label': 'Statut de rénovation'
        },
        color_discrete_map={'Rénové': '#636EFA', 'Non-rénové': '#EF553B'},
        category_orders={"renovated_label": ['Non-rénové', 'Rénové']}
    )

    # 7. Personnaliser le style
    fig.update_layout(
        barmode='overlay',
        bargap=0,
        showlegend=True,
        legend=dict(
            title="Statut des bâtiments",
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
        xaxis={'tickangle': -45} # Pivoter les étiquettes pour éviter le chevauchement
    )
    
    # Appliquer une opacité pour une meilleure lisibilité en mode 'overlay'
    for trace in fig.data:
        if trace.name == 'False': trace.marker.opacity = 0.7
        if trace.name == 'True': trace.marker.opacity = 0.9

    for frame in fig.frames:
        for trace in frame.data:
            if trace.name == 'False': trace.marker.opacity = 0.7
            if trace.name == 'True': trace.marker.opacity = 0.9

    return fig

# =========================