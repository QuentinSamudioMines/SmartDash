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
    # Tri par consommation par m²
    df_tri_m2 = df.copy()
    df_tri_m2["conso_m2"] = df_tri_m2["Consommation par m² par an (en kWh/m².an)_basic"]
    df_tri_m2 = df_tri_m2.sort_values(by="conso_m2", ascending=False).reset_index(drop=True)

    # Tri par consommation totale
    df_tri_MWh = df.copy()
    df_tri_MWh = df_tri_MWh.sort_values(by="total_energy_consumption_basic", ascending=False).reset_index(drop=True)

    # Stratégie aléatoire
    df_random = df.copy()
    df_random = df_random.sample(frac=1, random_state=42).reset_index(drop=True)

    return {
        "Consommation par m² (kWh/an)": df_tri_m2,         # clair et court
        "Consommation totale (MWh/an)": df_tri_MWh,       # simple et direct
        "Ordre aléatoire": df_random,              # reste simple
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
    
    # === NOUVEAU : Dictionnaire par type de bâtiment ET vecteur ===
    conso_par_type_et_vecteur = {
        "Résidentiel": {vec: [] for vec in vecteurs_energie},
        "Tertiaire": {vec: [] for vec in vecteurs_energie}
    }
    emissions_par_type_et_vecteur = {
        "Résidentiel": {vec: [] for vec in vecteurs_energie},
        "Tertiaire": {vec: [] for vec in vecteurs_energie}
    }
    
    n_annees = len(scenario_temporel)
    
    # === Construction des masques
    mask_res = df["UseType"] == "LOGEMENT"
    mask_tertiaire = df["UseType"] != "LOGEMENT" # adapte si besoin

    # Pré-calcule les index triés (pour stabilité dans le temps)
    df_res_sorted = df[mask_res].reset_index(drop=True)
    df_tert_sorted = df[mask_tertiaire].reset_index(drop=True)

    n_res = len(df_res_sorted)
    n_tert = len(df_tert_sorted)

    # Normalisation des scénarios temporels pour chaque groupe
    scenario_res = scenario_temporel * (coverage_by_group.get("Résidentiel", 0) / 100)
    scenario_tert = scenario_temporel * (coverage_by_group.get("Tertiaire", 0) / 100)

    for i_annee in range(n_annees):
        # === Sélection des bâtiments rénovés à cette année ===
        n_reno_res = int(scenario_res[i_annee] * n_res)
        n_reno_tert = int(scenario_tert[i_annee] * n_tert)

        df_reno_res = df_res_sorted.iloc[:n_reno_res]
        df_reno_tert = df_tert_sorted.iloc[:n_reno_tert]
        df_non_reno_res = df_res_sorted.iloc[n_reno_res:]
        df_non_reno_tert = df_tert_sorted.iloc[n_reno_tert:]

        # === Dictionnaires pour stocker les consommations par type ===
        conso_vect = {vec: 0.0 for vec in vecteurs_energie}
        conso_res = {vec: 0.0 for vec in vecteurs_energie}
        conso_tert = {vec: 0.0 for vec in vecteurs_energie}

        # === Calcul des consommations par vecteur et par type ===
        for vec in vecteurs_energie:
            # RÉSIDENTIEL
            conso_reno_res = df_reno_res[df_reno_res["energie_imope"] == vec]["total_energy_consumption_renovated"].sum() / 1000
            conso_rest_res = df_non_reno_res[df_non_reno_res["energie_imope"] == vec]["total_energy_consumption_basic"].sum() / 1000
            conso_res[vec] = conso_reno_res + conso_rest_res
            
            # TERTIAIRE
            conso_reno_tert = df_reno_tert[df_reno_tert["energie_imope"] == vec]["total_energy_consumption_renovated"].sum() / 1000
            conso_rest_tert = df_non_reno_tert[df_non_reno_tert["energie_imope"] == vec]["total_energy_consumption_basic"].sum() / 1000
            conso_tert[vec] = conso_reno_tert + conso_rest_tert
            
            # TOTAL (comme avant)
            conso_vect[vec] = conso_res[vec] + conso_tert[vec]

        # === Substitutions dynamiques (appliquées sur les totaux puis redistribuées) ===
        for source_vec, target_vec, percentage in substitutions:
            progression_scenario = scenario_temporel[i_annee] / scenario_temporel[-1]  # progressivité
            effective_substitution_rate = (percentage / 100) * progression_scenario

            # Calcul de la substitution sur le total
            conso_vect_avant = conso_vect.copy()
            conso_vect = substitute_energy(
                conso_vect,
                source_vec=source_vec,
                target_vec=target_vec,
                year_index=i_annee,
                total_years=n_annees,
                max_substitution_rate=effective_substitution_rate,
                efficiency_map=efficiency_map
            )
            
            # Redistribution proportionnelle de la substitution par type de bâtiment
            if conso_vect_avant[source_vec] > 0:
                ratio_res = conso_res[source_vec] / conso_vect_avant[source_vec]
                ratio_tert = conso_tert[source_vec] / conso_vect_avant[source_vec]

                # Application de la substitution proportionnellement
                delta_source = conso_vect[source_vec] - conso_vect_avant[source_vec]
                # Remplacez la ligne 184 par :
                delta_target = conso_vect.get(target_vec, 0.0) - conso_vect_avant.get(target_vec, 0.0)

                if target_vec not in conso_res:
                    conso_res[target_vec] = 0.0
                if target_vec not in conso_tert:
                    conso_tert[target_vec] = 0.0
                
                conso_res[source_vec] += delta_source * ratio_res
                conso_res[target_vec] += delta_target * ratio_res
                
                conso_tert[source_vec] += delta_source * ratio_tert
                conso_tert[target_vec] += delta_target * ratio_tert

        # === Stockage des résultats ===
        for vec in vecteurs_energie:
            # Stockage global (comme avant)
            conso_par_vecteur[vec].append(conso_vect[vec])
            facteur = facteurs_carbone.get(vec, np.zeros(n_annees))[i_annee]
            emissions = conso_vect[vec] * facteur
            emissions_par_vecteur[vec].append(emissions)
            
            # NOUVEAU : Stockage par type de bâtiment
            conso_par_type_et_vecteur["Résidentiel"][vec].append(conso_res[vec])
            conso_par_type_et_vecteur["Tertiaire"][vec].append(conso_tert[vec])
            
            emissions_res = conso_res[vec] * facteur
            emissions_tert = conso_tert[vec] * facteur
            emissions_par_type_et_vecteur["Résidentiel"][vec].append(emissions_res)
            emissions_par_type_et_vecteur["Tertiaire"][vec].append(emissions_tert)
    
    # === Marquer le statut de rénovation à 2050 ===
    n_reno_res_final = int(scenario_res[-1] * n_res)
    n_reno_tert_final = int(scenario_tert[-1] * n_tert)

    df_reno_final = pd.concat([
        df_res_sorted.iloc[:n_reno_res_final],
        df_tert_sorted.iloc[:n_reno_tert_final]
    ])
    df_reno_final = df_reno_final.copy()
    df_reno_final["etat_renovation_2050"] = "Rénové"
    df_reno_final["conso_finale_2050"] = df_reno_final["total_energy_consumption_renovated"] # en kWh

    df_non_reno_final = pd.concat([
        df_res_sorted.iloc[n_reno_res_final:],
        df_tert_sorted.iloc[n_reno_tert_final:]
    ])
    df_non_reno_final = df_non_reno_final.copy()
    df_non_reno_final["etat_renovation_2050"] = "Non rénové"
    df_non_reno_final["conso_finale_2050"] = df_non_reno_final["total_energy_consumption_basic"] # en kWh

    # === Concatenation finale ===
    df_simulated = pd.concat([df_reno_final, df_non_reno_final], ignore_index=True)

    return (df_simulated, conso_par_vecteur, emissions_par_vecteur, 
            conso_par_type_et_vecteur, emissions_par_type_et_vecteur)

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

def synthesize_results(df, conso_par_vecteur, emissions_par_vecteur, conso_par_type_et_vecteur, emissions_par_type_et_vecteur):
    """Calcule les bilans énergétiques et carbone avec répartition par type de bâtiment et vecteur énergétique.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données des bâtiments
        conso_par_vecteur (dict): Dictionnaire des consommations par vecteur énergétique
        emissions_par_vecteur (dict): Dictionnaire des émissions par vecteur énergétique
    
    Returns:
        dict: Dictionnaire contenant tous les indicateurs calculés
    """
    # === CALCULS GLOBAUX ===
    total_conso_2024 = sum([conso[0] for conso in conso_par_vecteur.values()])
    total_conso_2050 = sum([conso[-1] for conso in conso_par_vecteur.values()])
    total_emi_2024 = sum([em[0] for em in emissions_par_vecteur.values()])
    total_emi_2050 = sum([em[-1] for em in emissions_par_vecteur.values()])
    
    # === RÉPARTITION PAR VECTEUR ÉNERGÉTIQUE ===
    energy_breakdown = {
        "by_energy": {
            "conso_2024": {energy: conso[0] for energy, conso in conso_par_vecteur.items()},
            "conso_2050": {energy: conso[-1] for energy, conso in conso_par_vecteur.items()},
            "emissions_2024": {energy: em[0] for energy, em in emissions_par_vecteur.items()},
            "emissions_2050": {energy: em[-1] for energy, em in emissions_par_vecteur.items()},
        }
    }
    
    # Calcul des parts relatives
    for metric in ["conso_2024", "conso_2050", "emissions_2024", "emissions_2050"]:
        total = sum(energy_breakdown["by_energy"][metric].values())
        energy_breakdown["by_energy"][f"{metric}_pct"] = {
            energy: (value / total * 100) if total > 0 else 0
            for energy, value in energy_breakdown["by_energy"][metric].items()
        }
    
    # === RÉPARTITION PAR TYPE DE BÂTIMENT (si city_data disponible) ===
    building_breakdown = {}
    if conso_par_type_et_vecteur and emissions_par_type_et_vecteur:
        # Calcul des totaux par type de bâtiment
        residential_conso_2024 = sum([conso[0] for conso in conso_par_type_et_vecteur["Résidentiel"].values()])
        residential_conso_2050 = sum([conso[-1] for conso in conso_par_type_et_vecteur["Résidentiel"].values()])
        tertiary_conso_2024 = sum([conso[0] for conso in conso_par_type_et_vecteur["Tertiaire"].values()])
        tertiary_conso_2050 = sum([conso[-1] for conso in conso_par_type_et_vecteur["Tertiaire"].values()])
        
        residential_emi_2024 = sum([em[0] for em in emissions_par_type_et_vecteur["Résidentiel"].values()])
        residential_emi_2050 = sum([em[-1] for em in emissions_par_type_et_vecteur["Résidentiel"].values()])
        tertiary_emi_2024 = sum([em[0] for em in emissions_par_type_et_vecteur["Tertiaire"].values()])
        tertiary_emi_2050 = sum([em[-1] for em in emissions_par_type_et_vecteur["Tertiaire"].values()])
        
        # Calcul des totaux pour les pourcentages
        total_conso_2024_check = residential_conso_2024 + tertiary_conso_2024
        total_conso_2050_check = residential_conso_2050 + tertiary_conso_2050
        total_emi_2024_check = residential_emi_2024 + tertiary_emi_2024
        total_emi_2050_check = residential_emi_2050 + tertiary_emi_2050
        
        building_breakdown = {
            "by_building_type": {
                "residential": {
                    "conso_2024": residential_conso_2024,
                    "conso_2050": residential_conso_2050,
                    "emissions_2024": residential_emi_2024,
                    "emissions_2050": residential_emi_2050,
                    "share_conso_2024": (residential_conso_2024 / total_conso_2024_check * 100) if total_conso_2024_check > 0 else 0,
                    "share_conso_2050": (residential_conso_2050 / total_conso_2050_check * 100) if total_conso_2050_check > 0 else 0,
                    "share_emissions_2024": (residential_emi_2024 / total_emi_2024_check * 100) if total_emi_2024_check > 0 else 0,
                    "share_emissions_2050": (residential_emi_2050 / total_emi_2050_check * 100) if total_emi_2050_check > 0 else 0,
                    "reduction_conso_pct": ((residential_conso_2024 - residential_conso_2050) / residential_conso_2024 * 100) if residential_conso_2024 > 0 else 0,
                    "reduction_emissions_pct": ((residential_emi_2024 - residential_emi_2050) / residential_emi_2024 * 100) if residential_emi_2024 > 0 else 0
                },
                "tertiary": {
                    "conso_2024": tertiary_conso_2024,
                    "conso_2050": tertiary_conso_2050,
                    "emissions_2024": tertiary_emi_2024,
                    "emissions_2050": tertiary_emi_2050,
                    "share_conso_2024": (tertiary_conso_2024 / total_conso_2024_check * 100) if total_conso_2024_check > 0 else 0,
                    "share_conso_2050": (tertiary_conso_2050 / total_conso_2050_check * 100) if total_conso_2050_check > 0 else 0,
                    "share_emissions_2024": (tertiary_emi_2024 / total_emi_2024_check * 100) if total_emi_2024_check > 0 else 0,
                    "share_emissions_2050": (tertiary_emi_2050 / total_emi_2050_check * 100) if total_emi_2050_check > 0 else 0,
                    "reduction_conso_pct": ((tertiary_conso_2024 - tertiary_conso_2050) / tertiary_conso_2024 * 100) if tertiary_conso_2024 > 0 else 0,
                    "reduction_emissions_pct": ((tertiary_emi_2024 - tertiary_emi_2050) / tertiary_emi_2024 * 100) if tertiary_emi_2024 > 0 else 0
                }
            }
        }
    
    # === INDICATEURS DE PERFORMANCE ===
    intensity_2024 = (total_emi_2024 / total_conso_2024) * 1000 if total_conso_2024 > 0 else 0  # kgCO2/MWh
    intensity_2050 = (total_emi_2050 / total_conso_2050) * 1000 if total_conso_2050 > 0 else 0
    
    # === RASSEMBLEMENT DES RÉSULTATS ===
    results = {
        # Totaux globaux
        "Consommation 2024 (MWh)": total_conso_2024,
        "Consommation 2050 (MWh)": total_conso_2050,
        "Réduction (MWh)": total_conso_2024 - total_conso_2050,
        "Réduction (%)": 100 * (total_conso_2024 - total_conso_2050) / total_conso_2024 if total_conso_2024 > 0 else 0,
        
        # Émissions
        "Émissions 2024 (tCO₂)": total_emi_2024,
        "Émissions 2050 (tCO₂)": total_emi_2050,
        "Réduction émissions (tCO₂)": total_emi_2024 - total_emi_2050,
        "Réduction émissions (%)": 100 * (total_emi_2024 - total_emi_2050) / total_emi_2024 if total_emi_2024 > 0 else 0,
        
        # Intensité carbone
        "Intensité carbone 2024 (kgCO₂/MWh)": intensity_2024,
        "Intensité carbone 2050 (kgCO₂/MWh)": intensity_2050,
        "Réduction intensité (%)": 100 * (intensity_2024 - intensity_2050) / intensity_2024 if intensity_2024 > 0 else 0,
        
        # Répartitions
        "energy_breakdown": energy_breakdown,
        "building_breakdown": building_breakdown,
        
        # Métadonnées
        "energy_vectors": list(conso_par_vecteur.keys()),
        "years": len(next(iter(conso_par_vecteur.values()))) if conso_par_vecteur else 0,
    }
    
    return results

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
        title=dict(
            text=title,
            x=0.5,  # centré
            xanchor="center",
            yanchor="top"
        ),
        xaxis_title="Année",
        yaxis_title="Consommation (MWh)",
        hovermode="x unified",
        autosize=True,
        margin=dict(l=40, r=40, b=40, t=80),  # marges suffisantes pour le titre
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
        title=dict(
            text=title,
            x=0.5,  # centré horizontalement
            xanchor="center",
            yanchor="top"
        ),
        xaxis_title="Année",
        yaxis_title="Émissions (tCO₂)",
        hovermode="x unified",
        autosize=True,
        margin=dict(l=40, r=40, b=40, t=80),  # marge haute suffisante
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
        title=dict(
            text=f"Émissions cumulées de CO₂ (2024-2050) - Scénario {scenario_name}",
            x=0.5,  # centré
            xanchor="center",
            yanchor="top"
        ),
        xaxis_title="Année",
        yaxis_title="Émissions cumulées (tonnes CO₂)",
        hovermode="x unified",
        legend_title="Type d'énergie",
        autosize=True,
        margin=dict(l=40, r=40, b=40, t=80),  # marge haute pour ne pas couper le titre
        height=600
    )
    
    return fig

def add_energie_types(communes):
    """
    Construit le DataFrame part5 pour une ou plusieurs communes.
    
    Args:
        communes (list[str]): liste des noms de communes
    
    Returns:
        pd.DataFrame: lignes part5 associées aux communes
    """
    base = pd.DataFrame({
        "total_energy_consumption_basic": [0, 0, 0, 0, 0, 0, 0],
        "Consommation par m² par an (en kWh/m².an)_basic": [0, 0, 0, 0, 0, 0, 0],
        "total_energy_consumption_renovated": [0, 0, 0, 0, 0, 0, 0],
        "Consommation par m² par an (en kWh/m².an)_renovated": [0, 0, 0, 0, 0, 0, 0],
        "energie_imope": ["PAC Air-Air", "PAC Air-Eau", "PAC Géothermique",
                          "PAC Air-Air", "PAC Air-Eau", "PAC Géothermique", "bio gaz"],
        "heating_efficiency": [3.0, 4.0, 5.0, 3.0, 4.0, 5.0, 0.87],
        "UseType": ["LOGEMENT", "LOGEMENT", "LOGEMENT", "Autre", "Autre", "Autre", "Autre"],
    })

    duplicated = []
    for com in communes:
        temp = base.copy()
        temp["NOM_COM"] = com
        duplicated.append(temp)

    return pd.concat(duplicated, ignore_index=True)


@st.cache_data
def load_sample_data():
    """Charge et concatène les parties du fichier city_file, en ajoutant part5 pour chaque commune."""
    part1 = pd.read_pickle("city_part1.pkl")
    part2 = pd.read_pickle("city_part2.pkl")
    part3 = pd.read_pickle("city_part3.pkl")
    part4 = pd.read_pickle("city_part4.pkl")

    # Concaténation des parties principales
    city = pd.concat([part1, part2, part3, part4])

    # Ajout de part5 pour toutes les communes
    communes = city['NOM_COM'].dropna().unique()
    part5_full = add_energie_types(communes)

    return pd.concat([city, part5_full], ignore_index=True)


def filter_data_by_selection(city_data: pd.DataFrame, usage_selection: str, selected_com: bool = False):
    """
    Filtre les données selon la sélection d'usage et optionnellement sur les bâtiments CUD.
    
    Args:
        city_data: DataFrame contenant toutes les données
        usage_selection: "Résidentiel uniquement" ou "Résidentiel + Tertiaire"
        cud_only: Si True, ne garde que les bâtiments CUD
    
    Returns:
        pd.DataFrame: Données filtrées
    """
    # Filtre sur la commune
    if selected_com != "Toutes les communes":
        filtered_data = city_data[city_data["NOM_COM"] == selected_com]
        communes = [selected_com]
    else:
        filtered_data = city_data.copy()
        communes = city_data["NOM_COM"].dropna().unique()

    # Filtre usage
    if usage_selection == "Résidentiel":
        filtered_data = filtered_data[filtered_data["UseType"] == "LOGEMENT"]
    elif usage_selection == "Tertiaire":
        filtered_data = filtered_data[filtered_data["UseType"] != "LOGEMENT"]
    elif usage_selection == "Résidentiel + Tertiaire":
        pass
    else:
        filtered_data =  filtre_forme_juridique(filtered_data, ["COMU"])#, "COM", "CCOM"])#,"CCAS", "COAG", "COLL",  "DEPT", "EP", "EPA", "EPIC", "ETAT"]) # Filtre invalide
        # Ajout de part5 correspondant aux communes sélectionnées
    part5_full = add_energie_types(communes)
    filtered_data = pd.concat([filtered_data, part5_full], ignore_index=True)

    return filtered_data
    
def filtre_forme_juridique(
    df: pd.DataFrame, liste_formes_juridiques
) -> pd.DataFrame:
    """
    Vérifie si chaque ligne d'un DataFrame contient une forme juridique présente dans la liste donnée.
    Si une ligne ne contient pas une forme juridique valide, elle est supprimée du DataFrame.

    Args:
        df (pandas.DataFrame): Le DataFrame à vérifier.
        liste_formes_juridiques (list): Liste des formes juridiques valides.

    Returns:
        pandas.DataFrame: Le DataFrame modifié sans les lignes contenant des formes juridiques non valides.
    """
    # Création d'un masque booléen pour chaque ligne du DataFrame
    masque_lignes = df.apply(
        lambda row: row["forme_juridique"] in liste_formes_juridiques, axis=1
    )

    # Sélection des lignes valides à partir du masque
    df_valide = df[masque_lignes]

    return df_valide

def get_building_consumption_distribution(df, coverage_vector, year_index):
    logging.debug(f"\n🧪 len(df) = {len(df)} | len(coverage_vector) = {len(coverage_vector)}")

    n_batiments = len(df)
    n_reno = int(np.round(np.sum(coverage_vector)))
    sorted_indices = np.argsort(coverage_vector)
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

def display_assumptions(heating_efficiency_map, electricity_carbone_factor, 
                       facteurs_carbone, annees):
    """Affiche la section des hypothèses et paramètres utilisés"""
    st.header(" Hypothèses et données de simulation")
    
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
        # === TABLEAU DES DONNÉES UTILISÉES ===
    st.subheader("Bases de données mobilisées")
    data_sources = [
        {"Nom": "BDTOPO", "Description": "Base topographique haute résolution (IGN)", 
         "Référence": '<a href="https://geoservices.ign.fr/bdtopo" target="_blank">IGN BDTOPO</a>'},
        {"Nom": "BAN+", "Description": "Base nationale des adresses géolocalisées", 
         "Référence": '<a href="https://adresse.data.gouv.fr" target="_blank">BAN+</a>'},
        {"Nom": "SIRENE", "Description": "Registre des entreprises et établissements", 
         "Référence": '<a href="https://www.sirene.fr" target="_blank">INSEE SIRENE</a>'},
        {"Nom": "DPE", "Description": "Diagnostic de performance énergétique", 
         "Référence": '<a href="https://www.data.gouv.fr/fr/datasets/dpe" target="_blank">DPE officiel</a>'},
        {"Nom": "Open Data Enedis", "Description": "Consommation électrique par secteur d'activité (IRIS)", 
         "Référence": '<a href="https://data.enedis.fr/explore/dataset/consommation-electrique-par-secteur-dactivite-iris/information/" target="_blank">Open Data DLE</a>'},
        {"Nom": "Open Data GRDF", "Description": "Données locales de consommation de gaz",
         "Référence": '<a href="hhttps://opendata.grdf.fr/explore/dataset/consommation-annuelle-de-gaz-par-iris-et-code-naf0/information/" target="_blank">Open Data GRDF</a>'},
        {"Nom": "BDNB", "Description": "Base nationale des bâtiments", 
         "Référence": '<a href="https://www.data.gouv.fr/datasets/base-de-donnees-nationale-des-batiments/" target="_blank">BDNB</a>'},
        {"Nom": "FCU", "Description": "Réseaux de chaleur/froid urbains", 
         "Référence": '<a href="https://www.data.gouv.fr/datasets/traces-des-reseaux-de-chaleur-et-de-froid/" target="_blank">France Chaleur Urbaine</a>'},
        {"Nom": "2009 EDT", "Description": "Enquête emploi du temps, comportements", 
         "Référence": '<a href="https://www.insee.fr/fr/metadonnees/source/serie/s1224" target="_blank">INSEE 2009 EDT</a>'},
        {"Nom": "Open-Meteo", "Description": "Données météo horaires/journalières", 
         "Référence": '<a href="https://open-meteo.com/en/docs/historical-forecast-api" target="_blank">Open-Meteo</a>'}
    ]
    df_data = pd.DataFrame(data_sources)

    # ⚠️ Trick : utiliser st.markdown avec to_html() pour rendre les liens actifs
    st.markdown(df_data.to_html(escape=False, index=False), unsafe_allow_html=True)
