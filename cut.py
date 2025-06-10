import pandas as pd

# Chargement des données
city = pd.read_pickle("city_file.pkl")

# Calcul de la taille de chaque découpe
n = len(city)
chunk_size = n // 3

# Découpage
city_part1 = city.iloc[:chunk_size]
city_part2 = city.iloc[chunk_size:2*chunk_size]
city_part3 = city.iloc[2*chunk_size:]

# Sauvegarde
city_part1.to_pickle("city_part1.pkl")
city_part2.to_pickle("city_part2.pkl")
city_part3.to_pickle("city_part3.pkl")

print("Découpage terminé : city_part1.pkl, city_part2.pkl, city_part3.pkl créés.")
