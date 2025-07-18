import logging
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

def simulate(df, coverage_by_group, scenario_temporel, vecteurs_energie, efficiency_map, substitutions):
    """
    Exécute la simulation multi-usages avec des taux de rénovation différenciés (résidentiel vs tertiaire).
    Applique aussi les substitutions d'énergie et calcule les consommations/émissions.
    
    Args :
        df : DataFrame avec les bâtiments
        coverage_by_group : dict comme {"Résidentiel": 30, "Tertiaire": 20}
        scenario_temporel : np.array ou liste des parts cumulées de rénovation sur 2024 to 2050 (somme à 1)
        vecteurs_energie : liste des vecteurs (ex : ["Electricité", "Fioul", ...])
        efficiency_map : dict des rendements par vecteur
        substitutions : liste de tuples (source_vec, target_vec, pourcentage_max)
    """
    
    conso_par_vecteur = {vec: [] for vec in vecteurs_energie}
    emissions_par_vecteur = {vec: [] for vec in vecteurs_energie}
    n_annees = len(scenario_temporel)
    
    # === Construction des masques
    mask_res = df["UseType"] == "LOGEMENT"
    mask_tertiaire = df["UseType"] != "LOGEMENT" # adapte si besoin

    # Pré-calcule les index triés (pour stabilité dans le temps)
    df_res_sorted = df[mask_res].sort_values(by="ID").reset_index(drop=True)
    df_tert_sorted = df[mask_tertiaire].sort_values(by="ID").reset_index(drop=True)

    n_res = len(df_res_sorted)
    n_tert = len(df_tert_sorted)

    # Normalisation des scénarios temporels pour chaque groupe
    scenario_res = scenario_temporel * (coverage_by_group.get("Résidentiel", 0) / 100)
    scenario_tert = scenario_temporel * (coverage_by_group.get("Tertiaire", 0) / 100)

    for i_annee in range(n_annees):
        # === Sélection des bâtiments rénovés à cette année ===
        n_reno_res = int(scenario_res[i_annee] * n_res)
        n_reno_tert = int(scenario_tert[i_annee] * n_tert)

        df_reno = pd.concat([
            df_res_sorted.iloc[:n_reno_res],
            df_tert_sorted.iloc[:n_reno_tert]
        ])
        df_non_reno = pd.concat([
            df_res_sorted.iloc[n_reno_res:],
            df_tert_sorted.iloc[n_reno_tert:]
        ])

        conso_vect = {vec: 0.0 for vec in vecteurs_energie}

        # === Calcul des consommations par vecteur
        for vec in vecteurs_energie:
            conso_reno = df_reno[df_reno["energie_imope"] == vec]["total_energy_consumption_renovated"].sum()  / 1000
            conso_rest = df_non_reno[df_non_reno["energie_imope"] == vec]["total_energy_consumption_basic"].sum() / 1000
            conso_vect[vec] = conso_reno + conso_rest

        # === Substitutions dynamiques
        for source_vec, target_vec, percentage in substitutions:
            progression_scenario = scenario_temporel[i_annee] / scenario_temporel[-1]  # progressivité
            effective_substitution_rate = (percentage / 100) * progression_scenario

            conso_vect = substitute_energy(
                conso_vect,
                source_vec=source_vec,
                target_vec=target_vec,
                year_index=i_annee,
                total_years=n_annees,
                max_substitution_rate=effective_substitution_rate,
                efficiency_map=efficiency_map
            )

        # === Stockage des résultats
        for vec in vecteurs_energie:
            conso_par_vecteur[vec].append(conso_vect[vec])
            facteur = facteurs_carbone.get(vec, np.zeros(n_annees))[i_annee]
            emissions = conso_vect[vec] * facteur
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
        height=800
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
        height=800
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
        height=800,
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
    part4 = pd.read_pickle("city_part4.pkl")

    #"PAC Air-Air" : 3, "PAC Air-Eau" : 3, "PAC Eau-Eau" : 4, "PAC Géothermique" : 5
    part5 = pd.DataFrame({
        "total_energy_consumption_basic": [0, 0, 0,0,0,0],
        "Consommation par m² par an (en kWh/m².an)_basic": [0, 0, 0,0, 0, 0],
        "total_energy_consumption_renovated": [0, 0, 0,0, 0, 0],
        "Consommation par m² par an (en kWh/m².an)_renovated": [0, 0, 0,0, 0, 0],
        "energie_imope": ["PAC Air-Air", "PAC Air-Eau", "PAC Géothermique","PAC Air-Air", "PAC Air-Eau", "PAC Géothermique"],
        "heating_efficiency": [3.0, 4.0, 5.0,3.0, 4.0, 5.0],
        "UseType": ["LOGEMENT", "LOGEMENT", "LOGEMENT","Autre", "Autre", "Autre"],
    })  # Ajout d'une ligne vide pour éviter les erreurs de concaténation

    city = pd.concat([part1, part2, part3,part4, part5], ignore_index=True)
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
    
def get_building_consumption_distribution(df, coverage_vector, year_index):
    logging.debug(f"\n🧪 len(df) = {len(df)} | len(coverage_vector) = {len(coverage_vector)}")

    n_batiments = len(df)
    n_reno = int(np.round(np.sum(coverage_vector)))
    sorted_indices = np.argsort(-coverage_vector)
    idx_reno = sorted_indices[:n_reno]
    idx_non_reno = sorted_indices[n_reno:]

    df_reno = df.iloc[idx_reno].copy()
    df_non_reno = df.iloc[idx_non_reno].copy()

    # 🔍 Debug détaillé
    logging.debug(f"\n🔎 Year index: {year_index}")
    logging.debug(f"🧮 Total buildings: {n_batiments} | Renovated: {n_reno} | Non-renovated: {n_batiments - n_reno}\n")

    # 🚨 DIAGNOSTIC: Vérifier si les données rénovées existent
    logging.debug("🔍 Diagnostic des données rénovées:")
    renovated_col = 'Consommation par m² par an (en kWh/m².an)_renovated'
    basic_col = 'Consommation par m² par an (en kWh/m².an)_basic'
    
    logging.debug(f"📊 Colonnes disponibles: {list(df.columns)}")
    logging.debug(f"📈 Valeurs rénovées non-nulles: {df[renovated_col].notna().sum()}/{len(df)}")
    logging.debug(f"📈 Valeurs rénovées > 0: {(df[renovated_col] > 0).sum()}/{len(df)}")
    logging.debug(f"📈 Valeurs basic non-nulles: {df[basic_col].notna().sum()}/{len(df)}")
    
    # 🚨 CORRECTION: Gérer les valeurs manquantes ou nulles
    if df[renovated_col].isna().all() or (df[renovated_col] == 0).all():
        logging.debug("⚠️  ATTENTION: Toutes les valeurs rénovées sont nulles ou 0!")
        logging.debug("🔧 Application d'une correction: utilisation d'un facteur de réduction sur les valeurs basic")
        
        # Facteur de réduction typique pour la rénovation (30-50% de réduction)
        reduction_factor = 0.6  # 40% de réduction
        
        # Calculer les valeurs rénovées à partir des valeurs basic
        df_reno_corrected = df_reno.copy()
        df_reno_corrected[renovated_col] = df_reno[basic_col] * reduction_factor
        
        reno_consumption = df_reno_corrected[renovated_col].fillna(0)
        logging.debug(f"✅ Valeurs rénovées corrigées - Min: {reno_consumption.min():.1f}, Max: {reno_consumption.max():.1f}, Mean: {reno_consumption.mean():.1f}")
    else:
        reno_consumption = df_reno[renovated_col].fillna(0)
    
    non_reno_consumption = df_non_reno[basic_col].fillna(0)

    logging.debug("\n🧪 Rénovés (Consommation par m² par an):")
    logging.debug(reno_consumption.describe())
    
    logging.debug("\n🧪 Non-rénovés (Consommation par m² par an - basic):")
    logging.debug(non_reno_consumption.describe())

    logging.debug("\n📉 Vecteur coverage (résumé):")
    logging.debug(f"Min: {coverage_vector.min():.3f}, Max: {coverage_vector.max():.3f}, Mean: {coverage_vector.mean():.3f}")
    logging.debug(f"📊 Extrait coverage_vector: {coverage_vector[:10]}...")

    logging.debug(f"\n📌 Indices rénovés (top {len(idx_reno)}): {idx_reno[:10]}...")
    logging.debug(f"📌 Indices non-rénovés (reste): {idx_non_reno[:10]}...")

    logging.debug("\n📊 Valeurs consommation rénovés (extrait):")
    logging.debug(reno_consumption.head())

    logging.debug("\n📊 Valeurs consommation non-rénovés (extrait):")
    logging.debug(non_reno_consumption.head())

    # DataFrame temporaire pour manipulation sans toucher à df
    df_temp = pd.DataFrame({
        'renovated': [True] * len(df_reno) + [False] * len(df_non_reno),
        'energy_type': pd.concat([df_reno['energie_imope'], df_non_reno['energie_imope']], ignore_index=True),
        'consumption_m2': pd.concat([reno_consumption, non_reno_consumption], ignore_index=True)
    })

    logging.debug("\n🧮 Distribution finale (avant clip):")
    logging.debug(df_temp['consumption_m2'].describe())

    df_temp['consumption_m2'] = df_temp['consumption_m2'].clip(lower=0, upper=20000)

    # ✅ Vérification finale
    reno_data = df_temp[df_temp['renovated'] == True]
    non_reno_data = df_temp[df_temp['renovated'] == False]
    
    logging.debug(f"\n✅ Résultat final:")
    logging.debug(f"📊 Rénovés - Count: {len(reno_data)}, Mean: {reno_data['consumption_m2'].mean():.1f}, Min: {reno_data['consumption_m2'].min():.1f}")
    logging.debug(f"📊 Non-rénovés - Count: {len(non_reno_data)}, Mean: {non_reno_data['consumption_m2'].mean():.1f}, Min: {non_reno_data['consumption_m2'].min():.1f}")

    # ✅ Résultat encapsulé proprement
    solution = {
        'consumption_m2': df_temp['consumption_m2'].values,
        'renovated': df_temp['renovated'].values,
        'energy_type': df_temp['energy_type'].values,
        'n_renovated': n_reno,
        'n_total': n_batiments
    }

    return solution


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
    """
    coverage_by_useType = st.session_state["coverage_rates"]  # ex: {"Résidentiel": 30, "Tertiaire": 20}

    # Créer une série avec "Résidentiel" si UseType == "LOGEMENT", sinon "Tertiaire"
    mapped_useType = df['UseType'].apply(lambda x: "Résidentiel" if x == "LOGEMENT" else "Tertiaire")

    # Maintenant on récupère le coverage correspondant
    coverage_series = mapped_useType.map(coverage_by_useType).fillna(0) / 100.0
    
    # coverage_matrix shape = (n_batiments, n_années)
    coverage_matrix = np.outer(coverage_series, scenario_temporel)  

    all_data = []
    for i, year in enumerate(annees):
        # Passer la colonne i = taux de couverture pour l'année i pour chaque bâtiment
        coverage_appliquee = coverage_matrix[:, i]
        dist_data = get_building_consumption_distribution(df, coverage_appliquee, year_index=i)

        year_df = pd.DataFrame({
            'consumption_m2': dist_data['consumption_m2'],
            'renovated': dist_data['renovated'],
            'year': year
        })
        all_data.append(year_df)

    full_df = pd.concat(all_data, ignore_index=True)

    # 2. Définir les classes (bins) manuellement
    bin_size = 10  # Taille de chaque classe
    max_val = 1200 #! Important pour l'affichage
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
        color_discrete_map={'Rénové': '#636EFA', 'Non-rénové': "#E30B0B"},
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