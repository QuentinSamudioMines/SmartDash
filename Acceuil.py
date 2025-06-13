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
### Contexte et Problématique
Dans le cadre de la transition énergétique et de la lutte contre le changement climatique, le secteur du bâtiment représente un levier d'action majeur. Les collectivités territoriales, en première ligne pour la mise en œuvre des politiques de rénovation énergétique, nécessitent des outils d'aide à la décision robustes pour orienter leurs stratégies à long terme.

La **Modélisation Énergétique Urbaine** (ou *Urban Building Energy Modeling*, UBEM) s'est imposée comme une approche scientifique de premier plan pour estimer et simuler la consommation énergétique du parc bâti à l'échelle d'une ville ou d'un quartier (Reinhart et al., 2013). Malgré un développement académique soutenu, le transfert de ces outils vers un usage opérationnel par les acteurs de la planification territoriale reste un défi majeur.

---
### Le Projet MAP 2050
Le projet de recherche appliquée **MAP 2050** vise à combler ce fossé en développant un démonstrateur UBEM spécifiquement conçu pour soutenir la planification énergétique. En s'appuyant sur le cas d'étude de la **Communauté Urbaine de Dunkerque**, ce projet explore comment la modélisation peut éclairer les décisions publiques en matière de rénovation énergétique et de décarbonation du parc immobilier.

#### Objectifs du démonstrateur :
Cet outil a été conçu pour permettre aux utilisateurs de :
- **Explorer et comparer des scénarios de transition énergétique** en faisant varier les stratégies de rénovation (rythme, profondeur) et les bouquets énergétiques.
- **Visualiser l'impact de ces scénarios** sur la consommation énergétique globale et les émissions de gaz à effet de serre à l'horizon 2050.
- **Analyser la distribution des performances énergétiques** du parc immobilier et son évolution au fil du temps.

---

### Positionnement Scientifique
Le projet MAP 2050 s'inscrit dans les travaux de recherche sur l'évaluation des politiques publiques (Nouvel, 2013), les scénarios de rénovation à grande échelle (Johari, 2024 ; Rit, 2022) et l'aide à la décision pour la planification énergétique urbaine (Tol, 2012).

Il cherche à adresser certaines **limites récurrentes** des approches UBEM, notamment :
- La sensibilité à la qualité et la disponibilité des données d'entrée.
- La calibration et la validation des modèles à l'échelle urbaine.
- L'intégration des dynamiques socio-économiques dans les scénarios de rénovation.

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
Pour plus d’informations ou pour une collaboration, veuillez contacter :
- Quentin Samudio - Ingénieur de recherche - quentin.samudio@minesparis.psl.eu           

---
""")
