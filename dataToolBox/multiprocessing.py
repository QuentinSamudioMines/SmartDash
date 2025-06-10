import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor


def parallel_dataframe_processing(df, n, func):
    """
    Divise une DataFrame en `n` parties, applique une fonction `func` en parallèle à chaque partie,
    puis fusionne les résultats pour revenir à l'état d'origine.

    Parameters:
    - df: La DataFrame à diviser et traiter.
    - n: Le nombre de parties en lesquelles diviser la DataFrame.
    - func: La fonction à appliquer à chaque sous-DataFrame.

    Returns:
    - La DataFrame recombinée avec les résultats du traitement parallèle.
    """
    # Diviser la DataFrame en `n` parties
    df_splits = np.array_split(df, n)

    # Fonction pour traiter une seule partie de la DataFrame
    def process_subframe(sub_df):
        return func(sub_df)

    # Utiliser ThreadPoolExecutor pour un traitement parallèle
    with ThreadPoolExecutor() as executor:
        # Exécuter la fonction `process_subframe` en parallèle sur chaque partie
        results = list(executor.map(process_subframe, df_splits))

    # Fusionner les résultats dans une seule DataFrame
    df_merged = pd.concat(results)

    return df_merged


def linear_dataframe_processing(df, n, func):
    """
    Divise une DataFrame en `n` parties, applique une fonction `func` en parallèle à chaque partie,
    puis fusionne les résultats pour revenir à l'état d'origine.

    Parameters:
    - df: La DataFrame à diviser et traiter.
    - n: Le nombre de parties en lesquelles diviser la DataFrame.
    - func: La fonction à appliquer à chaque sous-DataFrame.

    Returns:
    - La DataFrame recombinée avec les résultats du traitement parallèle.
    """
    # Diviser la DataFrame en `n` parties
    df_splits = np.array_split(df, n)
    results = []
    for i in range(len(df_splits)):
        results.append(func(df_splits[i]))

    # Fusionner les résultats dans une seule DataFrame
    df_merged = pd.concat(results)

    return df_merged


# Exemple d'utilisation :
# Supposons que nous avons une DataFrame `df` et que nous voulons appliquer une fonction qui ajoute 1 à chaque colonne.

if __name__ == "__main__":
    # Fonction exemple qui ajoute 1 à chaque colonne
    def add_one(sub_df):
        return sub_df + 1

    # Appel de la fonction de traitement parallèle
    df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [10, 20, 30, 40, 50]})
    df_para = parallel_dataframe_processing(df, 2, add_one)
    df_line = linear_dataframe_processing(df, 2, add_one)
    print(df_para)
    print(df_line)
