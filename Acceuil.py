import streamlit as st

st.set_page_config(
    page_title="SmartE - MAP 2050 Project",
    page_icon="images/logo.png",
    layout="wide"
)

# Titre principal
st.title("MAP 2050 : Modélisation énergétique des territoires")

# Présentation du projet
st.markdown("""
Bienvenue sur le **étude de la Communauté Urbaine de Duncerque pour le projet MAP 2050**.

Le **projet MAP 2050** est un projet de **recherche appliquée** qui s'inscrit dans le champ de la **modélisation énergétique urbaine (UBEM)**. Son objectif est de **soutenir la planification énergétique et climatique des territoires** grâce à des outils de simulation adaptés.

La **modélisation énergétique urbaine** est une approche développée depuis une dizaine d'années pour **estimer la consommation énergétique des bâtiments à l'échelle urbaine, souvent avec des données limitées** (Reinhart et al., 2013). Cet outil est précieux pour accompagner les politiques publiques locales, mais son usage opérationnel reste encore limité dans les processus de planification.

---

### Ce que vous pouvez faire avec cette application :
- **Analyser des scénarios de transition énergétique** basés sur des hypothèses de rénovation, de décarbonation ou d’évolution des usages.
- **Soutenir la décision publique** avec des analyses spatialisées et des indicateurs pertinents.

---

### Contexte scientifique :
La littérature sur les UBEM est abondante et couvre des sujets tels que :
- L'aide au choix énergétique urbain (Tol, 2012)
- La prédiction de la demande énergétique (Hu, 2022)
- Les scénarios de rénovation à large échelle (Johari, 2024 ; Rit, 2022)
- L'évaluation des politiques publiques (Nouvel, 2013)

Cependant, plusieurs **limites persistent** :
- La qualité et la disponibilité des données d'entrée
- La complexité et le temps de calcul des modèles
- La difficulté de calibration et de validation à grande échelle

L'émergence des **open data** constitue une opportunité pour améliorer la précision et la robustesse des UBEM (Labetski et al., 2023).

---

### L’outil Smart-E  :

Le projet MAP 2050 s’appuie sur **Smart‑E**, une plateforme de simulation énergétique territoriale **développée par les Mines Paris – PSL**. Smart‑E permet de :
- Générer automatiquement un stock de bâtiments à partir de données open source.
- Modéliser les consommations du parc résidentiel et tertiaire.
- Simuler des scénarios de décarbonation (pompe à chaleur, isolation, autoconsommation).
- Fournir des indicateurs de performance, de flexibilité et de décarbonation à l’échelle d’un territoire.

MAP 2050 vise à proposer des **recommandations pratiques et des applications concrètes pour les collectivités** en exploitant les fonctionnalités de Smart‑E.

---

### Contact :
Pour plus d’informations ou pour une collaboration, veuillez contacter l’équipe MAP 2050 :
- Quentin Samudio - Ingénieur de recherche - quentin.samudio@minesparis.psl.eu           

---
""")
