import pandas as pd

from engine.City import City
from utils.logger import duration_logging


@duration_logging
def prune_df(df: pd.DataFrame) -> None:
    """
    Prune a DataFrame by removing rows with NaN or 0 values in specified columns.

    Parameters:
        df (pd.DataFrame): The DataFrame to be pruned.

    Returns:
        pd.DataFrame: The pruned DataFrame.
    """
    df = df.copy()
    # Define columns where NaN or 0 values should be checked
    columns_to_check = [
        "surface_habitable",
        "Total_power",
    ]  # Add your specific column names here
    # Boolean mask to filter rows
    mask = (
        df[columns_to_check].notna().all(axis=1)
    )  # Filter out rows where any of the specified columns have NaN
    mask &= (df[columns_to_check] != 0).all(
        axis=1
    )  # Filter out rows where any of the specified columns are 0
    mask &= (
        df["surface_habitable"] >= 20
    )  # Si c'est inférieur, on fait l'hypothèse que ça ne nous interesse pas
    # Apply the mask to prune the DataFrame
    df = df[mask]

    return df


@duration_logging
def get_DPE_value(df: pd.DataFrame) -> None:
    """
    Calculate the DPE (Diagnostic de Performance Énergétique) values for a given DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the necessary columns for DPE calculation.

    Returns:
        pd.DataFrame: The input DataFrame with additional columns for DPE values.
    """
    df = df.copy()
    df["Consommation annuelle (en MWh/an)"] = df["Total_power"] / 1000
    df["Consommation par m² par an (en kWh/m².an)"] = (
        df["Total_power"] / df["surface_habitable"]
    ) / 1000
    return df


def get_tertiary_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters a pandas DataFrame to only include rows where the 'USAGE1' column is neither 'Résidentiel' nor 'Indifférencié'.

    Parameters:
        df (pd.DataFrame): The input DataFrame to be filtered.

    Returns:
        pd.DataFrame: The filtered DataFrame containing only tertiary data.
    """
    df = df[(df["USAGE1"] != "Résidentiel") & (df["USAGE1"] != "Indifférencié")]
    return df


def get_all_cities(df: pd.DataFrame) -> dict[str, City]:
    """
    This function takes a pandas DataFrame as input, groups it by the 'NOM_COM' column,
    and returns a dictionary where each key is a city name and each value is a City object
    containing the corresponding DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the necessary columns for city grouping.

    Returns:
        dict[str, City]: A dictionary where each key is a city name and each value is a City object.
    """
    res = {}

    # On groupe le DataFrame par la colonne "NOM_COM" qui contient le nom des villes
    grouped = df.groupby("NOM_COM")

    # Pour chaque ville, on ajoute au dictionnaire le nom de la ville comme clé
    # et le DataFrame correspondant comme valeur
    for city, group in grouped:
        res[city] = City(city, group)

    return res


def remove_row_without_adress(df: pd.DataFrame) -> None:
    """
    Remove rows from a DataFrame that do not have a value in the "NOM_VOIE" column.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        None: The function modifies the input DataFrame in place.
    """
    df = df.dropna(subset=["NOM_VOIE"])
    return df


def get_dle_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtre les données du DataFrame pour conserver les lignes où il y a consommation
    d'électricité, de gaz ou de réseaux, et calcule la consommation totale en MWh/an.

    Parameters:
    df (pd.DataFrame): Le DataFrame original contenant les données de consommation.

    Returns:
    pd.DataFrame: Un nouveau DataFrame avec les lignes filtrées et une colonne
                  additionnelle pour la consommation totale.
    """
    # Create explicit copy at the start
    df_copy = df.copy()

    # Create boolean mask for filtering
    mask = (
        (df_copy["consommation_dle_elec"] != 0)
        | (df_copy["consommation_dle_gaz"] != 0)
        | (df_copy["consommation_dle_reseaux"] != 0)
    )

    # Apply filter using loc
    df_filtered = df_copy.loc[mask].copy()

    # Add new column using loc
    df_filtered.loc[:, "Donnée mesurée (en MWh/an)"] = (
        df_filtered["consommation_dle_elec"]
        + df_filtered["consommation_dle_gaz"]
        + df_filtered["consommation_dle_reseaux"]
    )

    return df_filtered
