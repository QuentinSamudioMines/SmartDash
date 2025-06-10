import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, Table, MetaData
import logging

# Configurer le logger
logging.basicConfig(
    level=logging.ERROR,  # Définit le niveau de logging
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Format des messages
)


class Database:
    def __init__(
        self,
        database: str,
        username: str = "root",
        password: str = "MhyH3ssKuEll3",
        hostname: str = "127.0.0.1",
    ):
        """
        Initialize the Database connection.

        :param username: MySQL username
        :param password: MySQL password
        :param hostname: MySQL server hostname
        :param database: MySQL database name
        """
        self.engine = create_engine(
            f"mysql+pymysql://{username}:{password}@{hostname}/{database}"
        )
        self.connection = self.engine.connect()
        self.metadata = MetaData()
        logging.info("Database connection established.")

    def transform_column_names(self, df):
        # Convert column names from Python timestamps to MySQL-compatible timestamps
        df.columns = [
            (
                self.python_timestamp_to_mysql_timestamp(col)
                if isinstance(col, datetime)
                else col
            )
            for col in df.columns
        ]
        return df

    def python_timestamp_to_mysql_timestamp(self, py_timestamp):
        # Convert Python timestamp to MySQL-compatible timestamp string
        return py_timestamp.strftime("%Y-%m-%d_%H_%M_%S")

    def store_dataframe(
        self, df: pd.DataFrame, table_name: str, if_exists: str = "replace"
    ) -> None:
        """
        Store a pandas DataFrame into a MySQL table.

        :param df: DataFrame to store
        :param table_name: Name of the table in the database
        :param if_exists: Behavior when the table already exists ('replace', 'append', 'fail')
        """
        df = self.transform_column_names(df)
        df.to_sql(table_name, self.engine, if_exists=if_exists, index=True)
        logging.info(
            f"DataFrame stored in table '{table_name}' with if_exists='{if_exists}'."
        )

    def retrieve_dataframe(self, table_name: str) -> pd.DataFrame:
        """
        Retrieve a pandas DataFrame from a MySQL table.

        :param table_name: Name of the table in the database
        :return: DataFrame with the data from the table
        """
        df = pd.read_sql_table(table_name, self.engine)
        logging.info(f"DataFrame retrieved from table '{table_name}'.")
        return df

    def insert_row(self, table_name: str, data: dict) -> None:
        """
        Insert a single row into a MySQL table.

        :param table_name: Name of the table in the database
        :param data: Dictionary representing the row to insert, with keys as column names
        """
        # Créer un objet MetaData pour contenir les informations sur la table
        metadata = MetaData()

        # Charger la table en utilisant les métadonnées
        table = Table(table_name, metadata, autoload_with=self.engine)

        # Préparer l'instruction d'insertion
        insert_stmt = table.insert().values(data)

        # Exécuter l'insertion
        with self.engine.begin() as connection:
            connection.execute(insert_stmt)

        logging.info(f"Row inserted into table '{table_name}'.")

    def close_connection(self) -> None:
        """
        Close the database connection.
        """
        self.connection.close()
        logging.info("Database connection closed.")


# Usage example
if __name__ == "__main__":
    # Replace with your own credentials
    database = "renove_batiment"

    db = Database(database)

    # Example DataFrame
    data = {"column1": [1, 2, 3], "column2": [4, 5, 6]}
    df = pd.DataFrame(data)

    # Store the DataFrame in the database
    db.store_dataframe(df, "example_table")

    # Insert a single row into the table
    new_row = {"column1": 4}
    db.insert_row("example_table", new_row)

    # Close the connection
    db.close_connection()
