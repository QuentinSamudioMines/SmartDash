import pandas as pd

# Chargement des données
city = pd.read_pickle("city_file.pkl")

survivor = ["total_energy_consumption_basic", "UseType","Consommation par m² par an (en kWh/m².an)_basic", "total_energy_consumption_renovated","Consommation par m² par an (en kWh/m².an)", "energie_imope", "heating_efficiency"]

# Filtrer et ne garder que la colonne 'survivor'
city_survivor = city[survivor]

# Calcul de la taille de chaque découpe
n = len(city_survivor)
chunk_size = n // 3

# Découpage
city_part1 = city_survivor.iloc[:chunk_size]
city_part2 = city_survivor.iloc[chunk_size:2*chunk_size]
city_part3 = city_survivor.iloc[2*chunk_size:]

# Sauvegarde des découpes
city_part1.to_pickle("city_part1.pkl")
city_part2.to_pickle("city_part2.pkl") 
city_part3.to_pickle("city_part3.pkl")

print("Découpage terminé : city_part1.pkl, city_part2.pkl, city_part3.pkl créés.")

