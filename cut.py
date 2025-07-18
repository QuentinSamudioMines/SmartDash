import pandas as pd

# Chargement des données
city1 = pd.read_pickle("11p1_renovated_analyse.pkl")
city2 = pd.read_pickle("11p2_renovated_analyse.pkl")

city = pd.concat([city1, city2], ignore_index=True)
print(list(city.columns))
survivor = ["ID","geometry", "surface_habitable","Consommation annuelle (en MWh/an)_basic" ,"Consommation par m² par an (en kWh/m².an)","UseType","CODE_IRIS","NOM_COM","Consommation par m² par an (en kWh/m².an)_basic", "Consommation annuelle (en MWh/an)", "energie_imope", "heating_efficiency"]

# Check if there are columns in survivor that are not in city
missing_in_city = set(survivor) - set(city.columns)
if missing_in_city:
    print(f"Columns in survivor that are not in city: {', '.join(missing_in_city)}")

# Count the number of Nan in each column
nan_counts = city.isna().sum()
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


