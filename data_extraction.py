# data_extraction.py        
import pandas as pd
import requests
import time
import boto3
from io import StringIO
from database_utils import DatabaseConnector
import tabula
from typing import Optional

class DataExtractor:
    def __init__(self):
        # Initialize DatabaseConnector instance and headers for API requests
        self.db_connector = DatabaseConnector()
        self.headers = {"x-api-key": "yFBQbwXe9J3sd6zWVAMrK6lcxxr0q1lr2PT6DDMX"}

    # RDS Table Extraction
    def extract_from_db(self, query: str) -> pd.DataFrame:
        """
        Extract data from the RDS database using a query and return the result as a DataFrame.
        :param query: SQL query to execute on the database.
        :return: DataFrame containing the queried data.
        """
        engine = self.db_connector.init_db_engine()
        if engine:
            return pd.read_sql(query, engine)
        return pd.DataFrame()

    def read_rds_table(self, table_name: str) -> pd.DataFrame:
        """
        Reads a table from the RDS database and returns it as a Pandas DataFrame.
        :param table_name: Name of the table to read from the database.
        :return: DataFrame containing the table's data.
        """
        try:
            engine = self.db_connector.engine
            tables = self.db_connector.list_db_tables()

            if table_name not in tables:
                return pd.DataFrame()

            query = f"SELECT * FROM {table_name}"
            return pd.read_sql(query, engine)
        except Exception:
            return pd.DataFrame()

    # PDF Extraction
    def retrieve_pdf_data(self, pdf_url: str) -> pd.DataFrame:
        """
        Extracts data from a PDF document hosted on a given URL and returns it as a DataFrame.
        :param pdf_url: URL of the PDF document.
        :return: DataFrame containing the extracted data.
        """
        try:
            dfs = tabula.read_pdf(pdf_url, pages="all", multiple_tables=True)
            return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        except Exception as e:
            print(f"Error extracting data from PDF: {e}")
            return pd.DataFrame()

    # API Extraction
    def list_number_of_stores(self, number_stores_endpoint: str) -> int:
        """
        Retrieves the number of stores from the API.
        :param number_stores_endpoint: API endpoint to get the number of stores.
        :return: Number of stores.
        """
        response = requests.get(number_stores_endpoint, headers=self.headers)
        if response.status_code == 200:
            data = response.json()
            return data.get("number_stores", 0)
        raise Exception(f"Failed to retrieve data: {response.status_code}, {response.text}")

    def retrieve_stores_data(self, retrieve_store_endpoint: str, num_stores: int) -> pd.DataFrame:
        """
        Retrieves store data from the API for each store and saves it in a pandas DataFrame.
        :param retrieve_store_endpoint: Endpoint template for store data.
        :param num_stores: Number of stores to retrieve data for.
        :return: DataFrame containing store data.
        """
        store_data = []
        for store_number in range(num_stores):
            store_url = retrieve_store_endpoint.format(store_number=store_number)
            response = requests.get(store_url, headers=self.headers)
            if response.status_code == 200:
                store_data.append(response.json())
            else:
                print(f"Failed to retrieve data for store {store_number}: {response.status_code}, {response.text}")
            time.sleep(0.2)

        return pd.DataFrame(store_data)

    # CSV Extraction from AWS
    def extract_from_s3(self, s3_path: str) -> Optional[pd.DataFrame]:
        """
        Extracts product CSV from an AWS S3 bucket using S3 address to return a DataFrame.
        :param s3_path: S3 path to the CSV file.
        :return: DataFrame containing the CSV data.
        """
        try:
            s3_client = boto3.client('s3')
            bucket_name, file_key = s3_path.replace("s3://", "").split("/", 1)
            obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
            data = obj['Body'].read().decode('utf-8')
            return pd.read_csv(StringIO(data), index_col=0)
        except Exception as e:
            print(f"Error extracting data from S3: {e}")
            return None

    # S3 AWS Extraction
    def extract_s3_datetime(self, url: str) -> Optional[pd.DataFrame]:
        """
        Extracts data from a public S3 URL containing JSON and returns it as a DataFrame.
        :param url: URL of the public S3 resource.
        :return: DataFrame containing the extracted data.
        """
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an error for bad responses
            return pd.DataFrame(response.json())
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            return None


    
    

