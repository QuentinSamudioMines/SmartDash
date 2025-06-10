import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import webbrowser
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import geopandas as gpd
import numpy as np
import folium
import logging
import shutil
import warnings
import folium
from folium import Marker
from shapely.geometry import Point
import plotly.graph_objects as go

from pyproj import Transformer

# Ignorer les avertissements de type UserWarning
warnings.filterwarnings("ignore", category=UserWarning)

from dataToolBox.pickelify import *
from dataToolBox.reader import *

# Configuration du logging
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def display_dataframe_in_tkinter(root, data: pd.DataFrame | gpd.GeoDataFrame) -> None:
    """
    Display a pandas DataFrame or geopandas GeoDataFrame in a Tkinter window.

    Parameters:
    - root: The root Tkinter window.
    - data: The DataFrame or GeoDataFrame to be displayed.

    Returns:
    None

    Functions:
    - show_next_100: Display the next 100 records.
    - export_to_excel: Export the data to an Excel file.
    - export_to_csv: Export the data to a CSV file.
    - export_to_pickle: Export the data to a pickle file.
    - export_metadata: Export metadata about the DataFrame to a text file.
    - update_treeview: Update the Treeview widget with the current data.
    - on_select: Handle the selection event in the Treeview widget.
    - validate_selection: Validate the selected row and display a map in a web browser.

    Logging:
    - Creates a child Tkinter window to display the DataFrame.
    - Logs the creation of the child window and the updates in the Treeview widget.
    - Logs the exportation of data to different file formats.
    - Logs the closing of the child window.
    """
    logging.info("Affichage du DataFrame dans une fenêtre Tkinter.")

    def show_next_100():
        logging.info("Affichage des 100 enregistrements suivants.")
        nonlocal start_index
        start_index += 100
        update_treeview()

    def export_to_excel():
        logging.info("Exportation des données vers Excel.")
        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")]
        )
        if file_path:
            data.to_excel(file_path, index=False)
            logging.info(f"Données exportées vers Excel : {file_path}")

    def export_to_csv():
        logging.info("Exportation des données vers un fichier CSV.")
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV files", "*.csv")]
        )
        if file_path:
            data.to_csv(file_path, index=False)
            logging.info(f"Données exportées vers le fichier CSV : {file_path}")

    def export_to_pickle():
        logging.info("Exportation des données vers un fichier pickle.")
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pkl", filetypes=[("Pickle files", "*.pkl")]
        )
        if file_path:
            data.to_pickle(file_path)
            logging.info(f"Données exportées vers le fichier pickle : {file_path}")

    # Todo : export_to_parquet()

    def export_to_parquet():
        logging.info("Exportation des données vers un fichier parquet.")
        file_path = filedialog.asksaveasfilename(
            defaultextension=".parquet", filetypes=[("Parquet files", "*.parquet")]
        )
        if file_path:
            data.to_parquet(file_path, index=False)
            logging.info(f"Données exportées vers le fichier parquet : {file_path}")

    def export_metadata():
        logging.info("Exportation des métadonnées dans un fichier texte.")
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt", filetypes=[("Text files", "*.txt")]
        )
        if file_path:
            with open(file_path, "w") as f:
                f.write("Métadonnées du DataFrame :\n\n")
                f.write(f"Nombre total de lignes : {len(data)}\n")
                f.write(f"Nombre total de colonnes : {len(data.columns)}\n\n")
                f.write("Noms des colonnes :\n")
                for col_name in data.columns:
                    f.write(f"- {col_name}\n")
                f.write("\nTypes de données des colonnes :\n")
                f.write(f"{data.dtypes}\n\n")
                f.write("Aperçu des premières lignes du DataFrame :\n")
                f.write(f"{data.head()}\n")
            logging.info(f"Métadonnées exportées vers le fichier texte : {file_path}")

    def update_treeview():
        logging.info("Mise à jour de Treeview avec les données actuelles.")
        tree.delete(*tree.get_children())
        for index, row in data.iloc[start_index : start_index + 100].iterrows():
            tree.insert("", "end", text=index, values=list(row))

    def on_select(event):
        selection = tree.selection()  # Récupère la sélection
        if selection:  # Vérifie si quelque chose est sélectionné
            selected_item = selection[0]  # Récupère l'élément sélectionné
            validate_selection_button.config(
                state="normal", command=lambda: validate_selection(selected_item)
            )
        else:
            # Gérer le cas où rien n'est sélectionné
            print("Aucun élément sélectionné")
        return

    def validate_selection(selected_item):
        selected_index = int(tree.item(selected_item, "text"))
        selected_row_data = data.iloc[selected_index]
        selected_geo_df = gpd.GeoDataFrame([selected_row_data])
        selected_geo_df.crs = "EPSG:4326"
        display_map_in_webbrowser(selected_geo_df)

    logging.info("Création d'une fenêtre enfant Tkinter pour afficher le DataFrame.")
    child_window = tk.Toplevel(root)
    child_window.title("Affichage du DataFrame")

    frame = ttk.Frame(child_window)
    frame.pack(fill="both", expand=True)

    tree = ttk.Treeview(frame)
    tree.pack(fill="both", expand=True)
    tree.heading("#0", text="Index")

    scrollbar_y = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    scrollbar_y.pack(side="right", fill="y")
    tree.configure(yscrollcommand=scrollbar_y.set)

    scrollbar_x = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
    scrollbar_x.pack(side="bottom", fill="x")
    tree.configure(xscrollcommand=scrollbar_x.set)

    tree["columns"] = list(data.columns)
    for column in data.columns:
        tree.heading(column, text=column)

    start_index = 0
    update_treeview()

    next_button = ttk.Button(child_window, text="Suivant", command=show_next_100)
    next_button.pack()

    export_to_parquet_button = ttk.Button(
        child_window, text="Exporter vers Parquet", command=export_to_parquet
    )
    export_to_parquet_button.pack()

    export_to_pickle_button = ttk.Button(
        child_window, text="Exporter vers Pickle", command=export_to_pickle
    )
    export_to_pickle_button.pack()

    export_to_excel_button = ttk.Button(
        child_window, text="Exporter vers Excel", command=export_to_excel
    )
    export_to_excel_button.pack()

    export_to_csv_button = ttk.Button(
        child_window, text="Exporter vers CSV", command=export_to_csv
    )
    export_to_csv_button.pack()

    export_metadata_button = ttk.Button(
        child_window, text="Exporter Métadonnées", command=export_metadata
    )
    export_metadata_button.pack()

    validate_selection_button = ttk.Button(
        child_window, text="Valider la sélection", state="disabled"
    )
    validate_selection_button.pack()

    tree.bind(
        "<<TreeviewSelect>>", on_select
    )  # Lie la fonction on_select à l'événement de sélection du Treeview

    # Message de logging lors de la fermeture de la fenêtre enfant
    child_window.protocol(
        "WM_DELETE_WINDOW",
        lambda: logging.info("Fermeture de la fenêtre enfant Tkinter.")
        or child_window.destroy(),
    )


def create_singleton_map(data: gpd.GeoDataFrame) -> str:
    """
    Create a singleton map for a given GeoDataFrame.

    Parameters:
        data (gpd.GeoDataFrame): The GeoDataFrame containing the geospatial data.

    Returns:
        str: The path to the saved singleton map.

    This function takes a GeoDataFrame containing geospatial data and creates a singleton map.
    A singleton map is a map centered on a specific location, with a marker at that location and
    the geospatial data displayed on the map. The function calculates the centroid of the
    GeoDataFrame, transforms the centroid coordinates from a specific coordinate reference
    system (CRS) to another (EPSG:2154 to EPSG:4326), and uses the transformed coordinates to
    create a map with a specified zoom level. It then adds a marker to the map at the
    transformed coordinates and adds the geospatial data to the map using the folium.GeoJson
    function. Finally, it saves the map temporarily and returns the path to the saved map.
    """
    centroid: Point = data.geometry.centroid.iloc[0]
    center_lat = centroid.y
    center_lon = centroid.x
    transformer = Transformer.from_crs("EPSG:2154", "EPSG:4326")
    lon, lat = transformer.transform(center_lon, center_lat)
    m = folium.Map([lon, lat], zoom_start=100)
    # Ajouter un marqueur à la position latlong
    folium.Marker(location=[lon, lat], popup="Selection").add_to(m)
    # Ajouter les données GeoJson à la carte
    # Identify columns where any element is a NumPy array
    data = drop_ndarray_columns(data)
    folium.GeoJson(data.to_json()).add_to(m)
    # Sauvegarder la carte temporairement
    singleton_path = "singleton_map.html"
    m.save(singleton_path)
    return singleton_path


def drop_ndarray_columns(df):
    """
    Drops columns from a DataFrame that contain any numpy.ndarray values.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame (or GeoDataFrame) to process.

    Returns:
    --------
    pandas.DataFrame
        A new DataFrame with columns dropped where any element is a numpy.ndarray.
    """
    # Identify columns where any element is an instance of np.ndarray
    cols_to_drop = [
        col
        for col in df.columns
        if df[col].apply(lambda x: isinstance(x, np.ndarray)).any()
    ]

    if cols_to_drop:
        print("Dropping columns:", cols_to_drop)
        df = df.drop(columns=cols_to_drop)

    return df


def create_sigleton_table(data: gpd.GeoDataFrame) -> str:
    """
    Creates a singleton table from a given GeoDataFrame.

    Parameters:
        data (gpd.GeoDataFrame): The GeoDataFrame containing the data to be displayed in the table.

    Returns:
        str: The HTML code for the singleton table.

    This function takes a GeoDataFrame and creates a singleton table by transposing the data and generating HTML code for the table. The table has two columns: one for the index names and one for the corresponding values. The function iterates over the transposed data and generates HTML rows for each index-value pair. The index names and value names are extracted from the data columns. The generated HTML code is then returned.
    """
    # Transposer le DataFrame pour l'affichage
    transposed_data = data.T

    # Préparer le HTML pour le tableau des informations du bâtiment
    building_info_html = """
    <table border=0 style="border: 1.2px solid #c6c6c6 !important; border-spacing: 2px; width: auto !important;">
        <thead>
            <tr>
                <th>{index_name}</th>
                <th>{value_name}</th>
            </tr>
        </thead>
        <tbody>
        {rows}
        </tbody>
    </table>
    """
    # Générer les lignes du tableau
    rows = ""
    for index, row in transposed_data.iterrows():
        rows += f"""
        <tr>
            <td>{index}</td>
            <td>{row.values[0]}</td>
        </tr>
        """
    # Remplacer les espaces réservés dans le template HTML
    index_name = data.columns[0]
    value_name = data.columns[1]
    return building_info_html.format(
        index_name=index_name, value_name=value_name, rows=rows
    )


def display_singleton_map(data: gpd.GeoDataFrame):
    """
    Affiche une carte avec un seul bâtiment dans un navigateur web.

    Cette fonction prend un GeoDataFrame contenant les données géographiques d'un bâtiment.
    Elle affiche une carte centrée sur ce bâtiment avec un marqueur à sa position.
    Un tableau scrollable contenant les détails du bâtiment est également affiché.

    Paramètres :
    - data : gpd.GeoDataFrame
        GeoDataFrame contenant les données géographiques du bâtiment.
    """
    logging.info("Affichage d'une carte dans un navigateur web.")
    map_path = create_singleton_map(data)
    building_info_html = create_sigleton_table(data)

    # Créer le HTML final
    template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Smart-magnifier</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background-color: #4183B6;  /* Blue background */
                color: black;
                margin: 0;
                padding: 0;
            }}
            .container {{
                display: flex;
                flex-direction: row;
                width: 100%;
            }}
            .map-container {{
                flex: 2;
                padding: 20px;
            }}
            .list-container {{
                flex: 1;
                padding: 20px;
                display: flex;
                flex-direction: column;
                background-color: #0056b3;  /* Darker blue for list container */
                border-radius: 10px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
            }}
            th, td {{
                padding: 5px;
                border: 1px solid #ddd;
                background-color: white;
                text-align: left;
            }}
            th {{
                font-weight: bold;
                background-color: #f2f2f2;
            }}
            .header {{
                display: flex;
                align-items: center;
                padding: 20px;
                color: #0056b3;
                background-color: white;
                border-bottom: 2px solid white;
            }}
            .header img {{
                height: 120px;
                margin-right: 20px;
            }}
        </style>
    </head>
    <body>
        <div class="header">
                <img src="logo.png" alt="Logo">
                <h1>Smart-magnifier</h1>
        </div>
        <div class="container">
            <div class="map-container">
                <h2>Localisation du Bâtiment</h2>
                <iframe src='{map_path}' style='width: 100%; height: 800px; border: none;'></iframe> 
            </div>
            <div class="list-container">
                <h2>Détails du Bâtiment</h2>
                {building_info_html} 
            </div>
        </div>
        
    </body>
    </html>
    """

    # Enregistrer le HTML final
    with open("singleton.html", "w", encoding="utf-8") as f:
        f.write(template)

    # Ouvrir le fichier HTML final dans le navigateur
    webbrowser.open("singleton.html")


def display_multiton_map(data: gpd.GeoDataFrame) -> None:
    """
    Display a multi-ton map using the given GeoDataFrame.

    Parameters:
        data (gpd.GeoDataFrame): The GeoDataFrame containing the geospatial data.

    Returns:
        None

    This function creates a Folium map centered at the default coordinates [51.0, 2.3] with a zoom start of 10. It adds the GeoJson data from the given GeoDataFrame to the map. The map is then saved temporarily as 'singleton_map.html'. The user is prompted to choose a file path to save the HTML file. If a file path is chosen, the temporary HTML file is moved to the specified location and opened in a web browser. The function logs the saved map file path.
    """
    # Centrer sur la couche
    data = ensure_crs(data)
    bounds = data.total_bounds
    center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]

    m = folium.Map(location=center, zoom_start=12) # Utilisez des valeurs par défaut si le DataFrame a plus d'un élément

    # Réparer les géométrie corrompus
    data["geometry"] = data.geometry.buffer(0)
    # Ajouter les données GeoJson à la carte
    data = drop_ndarray_columns(data)
    # Convertir les colonnes de type datetime en chaînes de caractères car elle ne sont pas sérializable en JSON
    for col in data.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]):
        data[col] = data[col].astype(str)

    # Ajouter les données GeoJson à la carte
    # Identifier les colonnes où tout élément est un scalaire ou une géométriemap_preload
    geo_cols = [col for col in data.columns if pd.api.types.is_scalar(data[col].iloc[0]) or col == 'geometry']
    clean_data = data[geo_cols].copy()
    geojson_str = clean_data.to_json()

    folium.GeoJson(geojson_str).add_to(m)

    # Sauvegarder la carte temporairement
    tmp = "temporary_map.html"
    logging.info(f"Sauvegarde de la carte temporaire : {tmp}")
    m.save(tmp)

    # Demander à l'utilisateur où sauvegarder le fichier HTML
    logging.info("Demande de l'emplacement de sauvegarde du fichier HTML.")
    file_path = filedialog.asksaveasfilename(
        defaultextension=".html", filetypes=[("HTML files", "*.html")]
    )

    if file_path:
        # Déplacer le fichier HTML temporaire vers l'emplacement spécifié par l'utilisateur
        shutil.move(tmp, file_path)
        logging.info(f"Carte sauvegardée en tant que HTML : {file_path}")
        logging.info("Ouverture du fichier HTML dans un navigateur web.")
        # Ouvrir le fichier HTML dans le navigateur web
        webbrowser.open(file_path)


def display_map_in_webbrowser(data: gpd.GeoDataFrame) -> None:
    """
    Display a map in a web browser.

    This function takes a GeoDataFrame containing geospatial data and displays it in a web browser.
    It creates a map with a default zoom level and adds the data to the map. If the DataFrame has only one element,
    it calls the display_singleton_map function to display a map with a marker at the element's location.
    Otherwise, it calls the display_multiton_map function to display a map with markers for all elements in the DataFrame.

    Parameters:
        data (gpd.GeoDataFrame): The GeoDataFrame containing the geospatial data.

    Returns:
        None

    Logs:
        - Affichage d'une carte dans un navigateur web.
    """
    logging.info("Affichage d'une carte dans un navigateur web.")
    # Créer une carte avec un zoom par défaut
    if len(data) == 1:  # Vérifier si le DataFrame a un seul élément
        display_singleton_map(data)
    else:
        display_multiton_map(data)


def ensure_crs(gdf: gpd.GeoDataFrame, target_crs="EPSG:4326") -> gpd.GeoDataFrame:
    """
    Ensure the GeoDataFrame is in the target CRS. If not, reproject it.
    Parameters:
    gdf (GeoDataFrame): The input GeoDataFrame.
    target_crs (str): The target CRS to transform to (default is "EPSG:3857").
    Returns:
    GeoDataFrame: The GeoDataFrame in the target CRS.
    """
    if gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)
    return gdf


def plotly_column(df, x_col, y_col, title=None, xlabel=None, ylabel=None):
    """
    Affiche une courbe interactive d'une colonne numérique en fonction d'une autre dans un DataFrame avec Plotly.

    :param df: pandas.DataFrame - Le DataFrame contenant les données.
    :param x_col: str - Le nom de la colonne à utiliser pour l'axe X.
    :param y_col: str - Le nom de la colonne à utiliser pour l'axe Y.
    :param title: str - (Optionnel) Titre de la courbe.
    :param xlabel: str - (Optionnel) Nom de l'axe X.
    :param ylabel: str - (Optionnel) Nom de l'axe Y.
    """
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(
            f"Les colonnes {x_col} ou {y_col} n'existent pas dans le DataFrame."
        )

    if not (df[x_col].dtype.kind in "biufc" and df[y_col].dtype.kind in "biufc"):
        raise TypeError(
            f"Les colonnes {x_col} et {y_col} doivent être de type numérique."
        )

    fig = px.line(
        df,
        x=x_col,
        y=y_col,
        title=title or f"Courbe de {y_col} en fonction de {x_col}",
        labels={x_col: xlabel or x_col, y_col: ylabel or y_col},
    )
    fig.update_layout(
        title_font_size=18,
        xaxis_title_font_size=14,
        yaxis_title_font_size=14,
        template="plotly_white",
    )
    fig.show()


def display_data(
    data: pd.DataFrame | gpd.GeoDataFrame,
    save_metadata: bool = False,
    file_for_save: str = "",
) -> None:  # display_data(root, data):
    """
    Display the given data in a Tkinter window with an option to choose between displaying it as a DataFrame or a GeoDataFrame.

    Args:
        data (pd.DataFrame | gpd.GeoDataFrame): The data to be displayed.

    Returns:
        None

    Logs:
        - Affichage des données.
        - Le type de données est un DataFrame.
        - Option sélectionnée : Tableau.
        - Le type de données est autre qu'un DataFrame.
        - Option sélectionnée : Tableau.
        - Option sélectionnée : Carte.
        - Fermeture de la fenêtre principale Tkinter.
        - Erreur lors de l'affichage des données : {str(e)}.
    """
    if save_metadata:
        logging.info("Exportation des métadonnées dans un fichier texte.")
        if file_for_save:
            with open(file_for_save, "w") as f:
                f.write("Métadonnées du DataFrame :\n\n")
                f.write(f"Nombre total de lignes : {len(data)}\n")
                f.write(f"Nombre total de colonnes : {len(data.columns)}\n\n")
                f.write("Noms des colonnes :\n")
                for col_name in data.columns:
                    f.write(f"- {col_name}\n")
                f.write("\nTypes de données des colonnes :\n")
                f.write(f"{data.dtypes}\n\n")
                f.write("Aperçu des premières lignes du DataFrame :\n")
                f.write(f"{data.head()}\n")
            logging.info(
                f"Métadonnées exportées vers le fichier texte : {file_for_save}"
            )
            return
    logging.info("Affichage des données.")
    try:
        root = tk.Tk()  # tk.Toplevel(root)
        root.title("SMartE - Affichage des données")

        # Définir une taille initiale plus grande
        root.geometry("400x100")

        option_var = tk.StringVar(root)
        option_var.set("Tableau")

        if type(data) == pd.DataFrame:
            logging.info("Le type de données est un DataFrame.")
            option_menu = tk.OptionMenu(root, option_var, "Tableau")
            option_menu.place(relx=0.05, rely=0.05, anchor="nw")

            def validate_option():
                selected_option = option_var.get()
                if selected_option == "Tableau":
                    logging.info("Option sélectionnée : Tableau.")
                    display_dataframe_in_tkinter(root, data)

        else:
            logging.info("Le type de données est autre qu'un DataFrame.")
            option_menu = tk.OptionMenu(root, option_var, "Tableau", "Carte")

            def validate_option():
                selected_option = option_var.get()
                if selected_option == "Tableau":
                    logging.info("Option sélectionnée : Tableau.")
                    display_dataframe_in_tkinter(root, data)
                elif selected_option == "Carte":
                    logging.info("Option sélectionnée : Carte.")
                    display_map_in_webbrowser(data)

        validate_button = tk.Button(root, text="Valider", command=validate_option)
        # Positionner le bouton en bas à gauche
        validate_button.place(relx=0.05, rely=0.95, anchor="sw")

        # Ajout d'un message de logging lors de la fermeture de la fenêtre principale
        root.protocol(
            "WM_DELETE_WINDOW",
            lambda: logging.info("Fermeture de la fenêtre principale Tkinter.")
            or root.quit(),
        )

        option_menu.place(relx=0.55, rely=0.05, anchor="nw")
        root.mainloop()
    except Exception as e:
        logging.error(f"Erreur lors de l'affichage des données : {str(e)}")
        messagebox.showerror(
            "Error", f"An error occurred while displaying data: {str(e)}"
        )


def select_file() -> str:
    """
    Open a file dialog to select a file.

    Returns:
        str: The path of the selected file, or None if no file is selected.
    """
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    return file_path


# Example usage
if __name__ == "__main__":
    logging.info("Starting application.")
    df = load_dataframe()  # Load your data here
    display_data(df)
    logging.info("Application finished.")
