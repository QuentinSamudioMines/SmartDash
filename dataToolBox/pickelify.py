import pickle
import logging
import time
import pandas as pd
import os
from tkinter import Tk, filedialog


def save_in_pkl(dataframe: pd.DataFrame, pkl_file_path: str = "") -> None:
    """
    Save a DataFrame to a pickle file.

    Parameters:
    dataframe (pd.DataFrame): The DataFrame to be saved.
    pkl_file_path (str, optional): The file path to save the pickle file. If not provided, a file dialog will open for the user to select the save location and file name.

    Returns:
    None
    """
    # If no file path is provided, open a file dialog for user input
    if pkl_file_path == "":
        root = Tk()
        root.withdraw()  # Hide the main window
        logging.info("Opening file dialog to select the save directory")
        print("Name the new pickle file")
        pkl_file_path = filedialog.asksaveasfilename(
            defaultextension=".pkl", filetypes=[("Pickle files", "*.pkl")]
        )
    else:
        pass

    # Check if the user has canceled the file selection
    if pkl_file_path:
        logging.info("Start saving pickle file")
        start = time.time()
        # Créez le répertoire s'il n'existe pas
        os.makedirs(os.path.dirname(pkl_file_path), exist_ok=True)
        # Save the DataFrame to the specified pickle file
        with open(pkl_file_path, "wb") as file:
            pickle.dump(dataframe, file)
        elapsed_time = time.time() - start
        logging.info(f"Object successfully saved in: {pkl_file_path}")
        logging.debug(f"Object saved in {elapsed_time:.2f} sec")
    else:
        logging.info("Operation canceled by the user.")


def load_pkl_file(pkl_file_path: str) -> pd.DataFrame:
    """
    Load a DataFrame from a pickle file.

    Parameters:
    pkl_file_path (str): The file path from which to load the pickle file.

    Returns:
    pd.DataFrame: The loaded DataFrame, or None if the operation was canceled by the user.
    """
    # Check if the user has provided a file path or has canceled the file selection
    if pkl_file_path:
        logging.info("Start loading pickle file")
        start = time.time()
        # Load the DataFrame from the specified pickle file
        with open(pkl_file_path, "rb") as f:
            loaded_data = pickle.load(f)
        elapsed_time = time.time() - start
        logging.info(f"Object successfully loaded from: {pkl_file_path}")
        logging.debug(f"Object loaded in {elapsed_time:.2f} sec")
        return loaded_data
    else:
        logging.info("Operation canceled by the user.")
        return None
