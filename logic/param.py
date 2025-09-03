# =========================
# Paramètres globaux
# =========================

import numpy as np


annees = np.arange(2024, 2051)
n_annees = len(annees)

# Scénarios de rénovation
scenarios_temporelles = {
    "Linéaire": np.linspace(0, 1, n_annees),
    "Rapide au début": np.array([1 * (1 - np.exp(-0.15 * i)) for i in range(n_annees)]),
    "Lent au début": np.array([1 * (1 - np.exp(-0.05 * i)) for i in range(n_annees)]),
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
    "bio gaz": np.full(n_annees, 0.180),
    "fioul": np.full(n_annees, 0.300),
    "bois": np.full(n_annees, 0.025),
    "chauffage urbain": np.linspace(0.127, 0.079, n_annees),
    "autre": np.full(n_annees, 0.150),
    "mixte": np.full(n_annees, 0.100),
    "PAC Air-Air": electricity_carbone_factor,
    "PAC Air-Eau": electricity_carbone_factor,
    "PAC Eau-Eau": electricity_carbone_factor,
    "PAC Géothermique": electricity_carbone_factor,
}
