# # # database_utils.py
import yaml
from sqlalchemy import create_engine, inspect
from typing import Optional, List
import pandas as pd

class DatabaseConnector:
    def __init__(self, use_local: bool = False):
        """
        Initialize the database connection.
        :param use_local: If True, connects to the local PostgreSQL (sales_data), otherwise to the read-only RDS database.
        """
        self.use_local = use_local
        self.db_creds = self._read_db_creds()
        self.engine = self._init_db_engine()

    def _read_db_creds(self) -> dict:
        """
        Reads database credentials from db_creds.yaml.
        :return: A dictionary containing database credentials.
        """
        try:
            with open("db_creds.yaml", "r") as file:
                return yaml.safe_load(file)
        except Exception as e:
            print(f"Error reading database credentials: {e}")
            return {}

    def _init_db_engine(self) -> Optional[object]:
        """
        Initializes a SQLAlchemy engine for database connection.
        :return: SQLAlchemy engine or None if the connection fails.
        """
        try:
            creds = self.db_creds
            connection_string = self._build_connection_string(creds)
            return create_engine(connection_string)
        except Exception as e:
            print(f"Error initializing database engine: {e}")
            return None

    def _build_connection_string(self, creds: dict) -> str:
        """
        Builds the connection string based on the credentials and connection preference (local or RDS).
        :param creds: A dictionary containing the database credentials.
        :return: A formatted database connection string.
        """
        if self.use_local:
            return f"postgresql://{creds['LOCAL_USER']}:{creds['LOCAL_PASSWORD']}@{creds['LOCAL_HOST']}:{creds['LOCAL_PORT']}/{creds['LOCAL_DATABASE']}"
        return f"postgresql://{creds['RDS_USER']}:{creds['RDS_PASSWORD']}@{creds['RDS_HOST']}:{creds['RDS_PORT']}/{creds['RDS_DATABASE']}"

    def list_db_tables(self) -> List[str]:
        """
        Lists all tables in the connected database.
        :return: A list of table names in the connected database.
        """
        try:
            return inspect(self.engine).get_table_names()
        except Exception as e:
            print(f"Error retrieving table names: {e}")
            return []

    def upload_to_db(self, df: pd.DataFrame, table_name: str) -> None:
        """
        Uploads a DataFrame to the connected database.
        :param df: DataFrame to upload.
        :param table_name: Name of the table to upload the data to.
        """
        if self.use_local and not df.empty:
            try:
                df.to_sql(table_name, self.engine, if_exists='replace', index=False)
            except Exception as e:
                print(f"Error uploading DataFrame to database: {e}")
