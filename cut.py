import logging
import pandas as pd

# Chargement des données
city10 = pd.read_pickle("11p1_renovated_3_analyse.pkl")
city20 = pd.read_pickle("11p2_renovated_3_analyse.pkl")

city1 = pd.read_pickle("12p1_renovated_analyse.pkl")
city2 = pd.read_pickle("12p2_renovated_analyse.pkl")
city3 = pd.read_pickle("12p3_renovated_analyse.pkl")
city4 = pd.read_pickle("12p4_renovated_analyse.pkl")
city5 = pd.read_pickle("12p5_renovated_analyse.pkl")
city6 = pd.read_pickle("12p6_renovated_analyse.pkl")
city7 = pd.read_pickle("12p7_renovated_analyse.pkl")
city8 = pd.read_pickle("12p8_renovated_analyse.pkl")

city_new = pd.concat([city1, city2, city3, city4, city5, city6, city7, city8], ignore_index=True)
nom_de_colonnes = [
    "code_insee_commune"
]
codes_Insee = [ 59359,
    59576,
    59110,
    59605,
    59082]

# This code allows you to display the cities one by one for verification of each code's correct assignment
# codes_communes_to_observe = [59210,59140]
# for i in codes_communes_to_observe:
#    df_filtre = filtre_commune(df, nom_de_colonnes, [i])
#    display_data(df_filtre)

def filtre_commune(df, nom_de_colonnes, codes_communes_liste):
    """
    Fonction pour filtrer un DataFrame en utilisant les noms de colonnes spécifiés et une liste de codes communes.

    Args:
    - df (DataFrame): Le DataFrame à filtrer.
    - nom_de_colonnes (list): Liste des noms de colonnes à filtrer.
    - codes_communes_liste (list): Liste des codes communes à inclure dans le filtre.

    Returns:
    - DataFrame: Le DataFrame filtré contenant uniquement les lignes correspondant aux codes communes spécifiés.

    * Liste des noms de colonnes à essayer #ban_code_commune_insee/ban_code_postal
    *nom_de_colonnes = ['code_commune_insee', 'INSEE_COM', 'ban_code_commune_insee', 'CODE_INSEE',"COMM"]

    * Ce que j'avais à l'origine était un mélange incorrect de code Postaux et Insee.
    * J'ai dû tatonné pour retrouver les bons. old = [59016, 59630, 59123, 59180, 59210, 59279, 59140, 59153, 59760, 59273, 59495, 59122, 59279, 59532, 59380, 59588, 59668]
    *codes_Insee_de_la_CUD = [59016, 59094, 59107, 59131, 59155, 59159, 59183, 59272, 59271, 59273, 59340, 59260, 59159, 59532, 59016, 59588, 59668]
    """
    df_filtre = pd.DataFrame()
    df = df.copy()
    for colonne in nom_de_colonnes:
        try:
            if df[colonne].dtype == "int64":
                # Filtrer directement si la colonne est de type int64
                df_filtre = df[df[colonne].isin(codes_communes_liste)]
            else:
                # Convertir les valeurs de la colonne en entiers si elles sont des chaînes de caractères
                df[colonne] = pd.to_numeric(
                    df[colonne], errors="coerce"
                )  # Convertir les non-convertibles en NaN
                df_filtre = df[df[colonne].isin(codes_communes_liste)]
        except KeyError:
            logging.error(f"La colonne '{colonne}' n'existe pas dans le DataFrame.")
            pass
    if df_filtre.empty:
        logging.info("Aucune ligne ne correspond aux critères de filtre.")
    return df_filtre

city30 = filtre_commune(city_new, nom_de_colonnes, codes_Insee)
print(f"Number of rows in city30: {len(city30)}")

city = pd.concat([city10, city20, city30], ignore_index=True)
print(list(city.columns))
survivor = ["ID","geometry", "surface_habitable","Consommation annuelle (en MWh/an)_basic", "forme_juridique","Consommation par m² par an (en kWh/m².an)","UseType","CODE_IRIS","NOM_COM","Consommation par m² par an (en kWh/m².an)_basic", "Consommation annuelle (en MWh/an)", "energie_imope", "heating_efficiency"]

# Check if there are columns in survivor that are not in city
missing_in_city = set(survivor) - set(city.columns)
if missing_in_city:
    print(f"Columns in survivor that are not in city: {', '.join(missing_in_city)}")

# Count the number of Nan in each column
nan_counts = city.loc[:, city.columns != "geometry"].isna().sum()

print("Number of NaN in each column:")
for col, count in nan_counts.items():
    print(f"- {col}: {count}")
# Filtrer et ne garder que la colonne 'survivor'
city_survivor = city[~city["UseType"].str.contains("INDUSTRIE")][survivor]



city_survivor["total_energy_consumption_basic"] = city_survivor["Consommation annuelle (en MWh/an)_basic"] * 1000
city_survivor["total_energy_consumption_renovated"] = city_survivor["Consommation annuelle (en MWh/an)"] * 1000
city_survivor["Consommation par m² par an (en kWh/m².an)_renovated"] = city_survivor["Consommation par m² par an (en kWh/m².an)"]
# Calcul de la taille de chaque découpe
n = len(city_survivor)
d = 4  # Nombre de découpes souhaitées
chunk_size = n // d

# Découpage
for i in range(d):
    start_index = i * chunk_size
    end_index = (i + 1) * chunk_size if i < d - 1 else n
    city_survivor.iloc[start_index:end_index].to_pickle(f"city_part{i+1}.pkl")

# Affichage du résultat
print(f"Découpage en {d} parties effectué. Chaque partie contient environ {chunk_size} lignes.")


