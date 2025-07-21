"""
# SmartDash - Simulation de rénovation énergétique et émissions carbone
# 
# Cette application permet de simuler l'impact des stratégies de rénovation énergétique 
# sur la consommation d'énergie et les émissions de carbone dans une ville.
# Elle permet aux utilisateurs de choisir différentes stratégies et scénarios de rénovation,
# et visualise les résultats sous forme de graphiques interactifs.
#
# Installation des dépendances requises :
# pip install streamlit plotly pandas numpy
#
# Commande pour lancer l'application :
# streamlit run Acceuil.py 
"""
import streamlit as st
import os

# Configuration de la page
if os.path.exists("images/logo.png"):
    st.set_page_config(
        page_title="SmartE - MAP 2050 Project",
        page_icon="images/logo.png",
        layout="wide"
    )
else:
    st.set_page_config(
        page_title="SmartE - MAP 2050 Project",
        layout="wide"
    )

# Titre principal
st.title("MAP 2050 : Modélisation énergétique des territoires")

# Présentation du projet
st.markdown("""
Bienvenue sur l’**étude de la Communauté Urbaine de Dunkerque pour le projet MAP 2050**.

Le **projet MAP 2050** est un projet de **recherche appliquée** qui s'inscrit dans le champ de la *Modélisation Énergétique Urbaine (UBEM)*. Son objectif est de **soutenir la planification énergétique et climatique des territoires** grâce à des outils de simulation adaptés.

---

### 🔍 Contexte et Problématique
Dans le cadre de la transition énergétique et de la lutte contre le changement climatique, le secteur du bâtiment représente un levier d'action majeur. Les collectivités territoriales, en première ligne pour la mise en œuvre des politiques de rénovation énergétique, nécessitent des outils d'aide à la décision robustes pour orienter leurs stratégies à long terme.

La *Modélisation Énergétique Urbaine* (UBEM) s'est imposée comme une approche scientifique de premier plan pour estimer et simuler la consommation énergétique du parc bâti à l'échelle d'une ville ou d'un quartier (Reinhart et al., 2013). Malgré un développement académique soutenu, le transfert de ces outils vers un usage opérationnel par les acteurs de la planification territoriale reste un défi majeur.

---

### 🧪 Le Projet MAP 2050
Le projet de recherche appliquée MAP 2050 vise à identifier les avantages et limites des UBEM pour la planification énergétique par une mise en application sur un cas concret. En s'appuyant sur le cas d'étude de la Communauté Urbaine de Dunkerque, ce projet explore comment la modélisation peut éclairer les décisions publiques en matière de rénovation énergétique et de décarbonation du parc immobilier.

#### 🎯 Objectifs du démonstrateur
Cet outil a été conçu pour permettre aux utilisateurs de :
- **Explorer et comparer des scénarios de transition énergétique** en faisant varier les stratégies de rénovation (rythme, profondeur) et les bouquets énergétiques.
- **Visualiser l'impact de ces scénarios** sur la consommation énergétique globale et les émissions de gaz à effet de serre à l'horizon 2050.
- **Analyser la distribution des performances énergétiques** du parc immobilier et son évolution au fil du temps.

#### 📌 Conclusions
Le démonstrateur illustre le type de résultats que l’on peut obtenir à partir de données publiques sur les bâtiments. Il montre l’apport des outils de simulation physique des bâtiments pour l’étude de scénarios avec quelques exemples, sans rechercher l’exhaustivité des potentialités offertes par l’outil. En revanche, la principale limite aujourd’hui est la fiabilité des données disponibles, notamment en ce qui concerne le parc de bâtiments tertiaires.

---

### 🛠️ L’outil Smart‑E

Le projet MAP 2050 s’appuie sur **Smart‑E**, une plateforme de simulation énergétique territoriale **développée par les Mines Paris – PSL**. Smart‑E permet de :
- Générer automatiquement un stock de bâtiments à partir de données open source.
- Modéliser les consommations du parc résidentiel et tertiaire.
- Simuler des scénarios de décarbonation (pompe à chaleur, isolation, autoconsommation).
- Fournir des indicateurs de performance, de flexibilité et de décarbonation à l’échelle d’un territoire.

MAP 2050 vise à proposer des **recommandations pratiques et des applications concrètes pour les collectivités** en exploitant les fonctionnalités de Smart‑E.

---

### ✅ Vérification du moteur physique : tests de type BESTEST

Avant d’être utilisé à l’échelle territoriale, le moteur physique de SmartE a été **validé par des tests de référence standardisés**, notamment les cas **BESTEST-EX**, reconnus pour l’évaluation des outils de simulation thermique.

Les résultats ci-dessous comparent SmartE à trois logiciels de référence : **EnergyPlus**, **SUNREL** et **DOE-2.1E**, sur plusieurs cas de test axés sur les transferts thermiques dans des bâtiments simples. Ces comparaisons confirment que le **comportement thermique simulé par SmartE est cohérent** avec les standards internationaux de modélisation.
""")

st.image("images/bestest_smarte.png", caption="Résultats des tests BESTEST-EX : comparatif SmartE vs références", use_column_width=True)

st.markdown("""
#### 📌 Interprétation des cas BESTEST

Les écarts observés sont jugés **acceptables** et confirment le bon fonctionnement du moteur physique de SmartE. Chaque cas représente un scénario spécifique de performance thermique :

- **200** : Cas de base servant de référence, bâtiment simple avec infiltration standard.
- **210** : Réduction des infiltrations d’air, amélioration de l’étanchéité.
- **220** : Rénovation thermique de la toiture (amélioration de l’isolation supérieure).
- **225** : Rénovation des murs (amélioration de l’isolation latérale).
- **240** : Modification du scénario d’occupation (hausse ou baisse de présence humaine).
- **250** : Pose de double vitrage (amélioration de la performance des menuiseries).

Les résultats de SmartE se situent dans les marges acceptées par le protocole de test, démontrant que les effets thermiques induits par chaque modification sont bien représentés.

---

### 📊 Validation de la simulation

La validation du modèle repose sur la comparaison entre les consommations simulées par **SmartE** et des **données de référence** issues de bases de données publiques officielles et récentes. Ces données couvrent les consommations énergétiques à l’échelle territoriale et sectorielle, permettant une analyse fine et fiable des écarts.

Les principales sources utilisées sont :

- [Consommation annuelle de gaz par département et code NAF - GRDF](https://opendata.grdf.fr/explore/dataset/consommation-annuelle-de-gaz-par-departement-et-code-naf/table/) : données détaillées de consommation de gaz naturel par secteur d’activité et territoire.
- [Consommation électrique par secteur d’activité à l’échelle IRIS - Enedis](https://data.enedis.fr/explore/dataset/consommation-electrique-par-secteur-dactivite-iris/table/?sort=annee) : consommation électrique détaillée par secteur d’activité et unité territoriale fine.

Cette validation a donc été réalisée **à la maille IRIS**, ce qui permet une analyse territorialisée des écarts avec notre référence.

Deux catégories principales de bâtiments ont été distinguées :
- **Logements**
- **Autres bâtiments** (tertiaires, services publics, etc.)

Les graphiques suivants comparent la consommation estimée par SmartE à la consommation de référence pour chaque secteur.
La ligne rouge représente la **parfaite correspondance (y = x)**, tandis que les lignes verte et bleue indiquent une marge de **±10 %**.

""")

col1, col2 = st.columns(2)

with col1:
    st.image("images/validation_logement.png", caption="Comparaison SmartE vs Référence - Logement", use_column_width=True)

with col2:
    st.image("images/validation_autres.png", caption="Comparaison SmartE vs Référence - Autres bâtiments", use_column_width=True)

st.markdown("""
### 🧾 Interprétation

- Une majorité de points se situent entre les lignes ±10 %, ce qui montre une bonne cohérence globale du modèle.
- Des écarts importants s’observent sur certaines IRIS, souvent liés à des **incertitudes sur la surface**, l’usage ou le système énergétique.
- Le modèle a tendance à **surestimer** les consommations, en moyenne de **30 %**, avec un **écart type également d’environ 30 %** sur les deux catégories de bâtiments (logement et autres).

Cette **surconsommation est intrinsèque à notre méthode de paramétrisation**, qui repose sur des hypothèses prudentes (notamment pour le tertiaire) afin de ne pas sous-estimer les consommations réelles. %, ce qui montre une bonne cohérence globale du modèle.

**Conclusion :** La validation montre que **SmartE** offre des estimations fiables à l’échelle territoriale, avec une **marge d’erreur acceptable pour un outil d’aide à la décision**. Des améliorations sont possibles via une meilleure qualification des données d'entrée.

---

### 📬 Contact
Pour plus d’informations ou pour une collaboration, veuillez contacter :
- Quentin Samudio - Ingénieur de recherche - quentin.samudio@minesparis.psl.eu

---
""")