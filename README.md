# Multinational Retail Data Centralisation
The Multinational Data Centralisation project focuses on transforming and analyzing large datasets from various sources. It leverages Pandas for data cleaning, generates a STAR-based database schema for efficient storage and access, and develops complex SQL queries for data analysis, offering a comprehensive solution from data acquisition to insightful decision-making.

This course helped me enhance my skills in data extraction from various database sources, cleaning data from multiple sources using Python, and uploading it to my local PostgreSQL database, 'sales_data'.

1. [Milestone 1](#milestone-1)
   - [Description](#description)
2. [Milestone 2](#milestone-2)
   - [Description](#description-1)
   - [User Data](#user-data)
   - [Card Details](#card-details)
   - [Store Details](#store-details)
   - [Product Details](#product-details)
   - [Order tables](#order-tables)
   - [Events Data](#events-data)
   - [File Structure](#file-structure-1)
3. [Milestone 3](#milestone-3)
   - [Description](#description-2)
   - [File Structure](#file-structure-2)
4. [Milestone 4](#milestone-4)
   - [Description](#description-3)
   - [File Structure](#file-structure-3)

## Milestone 1
Setting-up Github Environment.

### Description
This milestone introduces GitHub for tracking and saving code changes in a GitHub repo, which will be use throughout the project.

## Milestone 2
### Description
Extracting and cleaning data from various sources, including RDS tables, PDFs, APIs, and AWS S3 buckets. A 'DatabaseConnector' class was created to establish connections and upload the data into a local PostgreSQL database.

### User Data
#### DataConnector Class
```
from sqlalchemy import create_engine, inspect
import yaml

class DatabaseConnector:
    def __init__(self, use_local=False):
        """
        Initialize database connection.
        If use_local=True, connect to the local PostgreSQL (sales_data).
        """
        self.use_local = use_local
        self.db_creds = self.read_db_creds()
        self.engine = self.init_db_engine()

    def read_db_creds(self) -> dict:
        """Reads database credentials from db_creds.yaml."""
        try:
            with open("db_creds.yaml", "r") as file:
                return yaml.safe_load(file)
        except Exception:
            return {}

    def init_db_engine(self):
        """Initializes a SQLAlchemy engine for database connection."""
        try:
            creds = self.db_creds
            if self.use_local:
                return create_engine(
                    f"postgresql://{creds['LOCAL_USER']}:{creds['LOCAL_PASSWORD']}@{creds['LOCAL_HOST']}:{creds['LOCAL_PORT']}/{creds['LOCAL_DATABASE']}"
                )
            return create_engine(
                f"postgresql://{creds['RDS_USER']}:{creds['RDS_PASSWORD']}@{creds['RDS_HOST']}:{creds['RDS_PORT']}/{creds['RDS_DATABASE']}"
            )
        except Exception:
            return None

    def list_db_tables(self):
        """Lists all tables in the connected database."""
        try:
            return inspect(self.engine).get_table_names()
        except Exception:
            return []

    def upload_to_db(self, df, table_name: str):
        """Uploads a DataFrame to the connected database."""
        if self.use_local and not df.empty:
            try:
                df.to_sql(table_name, self.engine, if_exists='replace', index=False)
            except Exception:
                pass
```

#### DataExtractor Class 
```
import pandas as pd
from database_utils import DatabaseConnector

class DataExtractor:
    def __init__(self):
        self.db_connector = DatabaseConnector()  # Initialize DatabaseConnector instance

    def extract_from_db(self, query: str) -> pd.DataFrame:
        """
        Extract data from the RDS database using a query and return the result as a DataFrame.
        """
        engine = self.db_connector.init_db_engine()
        return pd.read_sql(query, engine) if engine else pd.DataFrame()

    def read_rds_table(self, table_name: str) -> pd.DataFrame:
        """
        Reads a table from the RDS database and returns it as a Pandas DataFrame.
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
```

#### DataCleaning Class
```
import pandas as pd
import numpy as np

class DataCleaning:
    def __init__(self):
        pass

    def remove_null(self, table):
        """Replaces 'NULL' with NaN and removes rows with missing values."""
        table = pd.DataFrame(table)  # Ensure table is a DataFrame
        table.replace("NULL", np.nan, inplace=True)  # Replace "NULL" strings with NaN
        table.dropna(inplace=True)  # Drop rows with NaN values
        table.reset_index(drop=True, inplace=True)  # Reset index after dropping rows
        return table

    def valid_date(self, table, date_column):
        """Cleans and converts a date column to datetime format, ensuring only valid dates remain."""
        if date_column in table.columns:
            # Identify alphanumeric (non-date) values and remove them
            non_date_mask = table[date_column].astype(str).str.match(r'^[A-Za-z0-9]+$', na=False)
            table = table[~non_date_mask]

            # Convert remaining values to datetime
            table[date_column] = pd.to_datetime(table[date_column], format="mixed", errors="coerce")

        return table
    
    def clean_user_data(self, table):
        """
        Cleans user data by:
        - Removing 'NULL' string values and replacing them with NaN
        - Dropping rows with NaN values
        - Ensuring 'join_date' is in datetime format, handling invalid dates
        """
        table = self.remove_null(table)  # Remove NULL values
        table = self.valid_date(table, 'join_date')  # Convert date column

        return table
```

### Card Details

#### DataExtractor Class 
```
import pandas as pd
import tabula 

class DataExtractor:

    def retrieve_pdf_data(self, pdf_url: str) -> pd.DataFrame:
        """
        Extracts data from a PDF document hosted on a given URL and returns it as a DataFrame.
        
        :param pdf_url: URL of the PDF document.
        :return: DataFrame containing the extracted data.
        """
        try:
            # Use tabula to extract all pages from the PDF
            dfs = tabula.read_pdf(pdf_url, pages="all", multiple_tables=True)

            # If multiple tables are returned, we concatenate them into a single DataFrame
            full_data = pd.concat(dfs, ignore_index=True)
            return full_data
        
        except Exception as e:
            print(f"Error extracting data from PDF: {e}")
            return pd.DataFrame()  # Return an empty DataFrame in case of an error
```

#### DataCleaning Class
```
class DataCleaning:
    def __init__(self):
        pass

    def check_null_values(self, table):
        """Returns the number of NULL values in each column."""
        null_counts = table.isnull().sum()
        return null_counts

    def check_duplicate_rows(self, table, card_column):
        """Returns the number of duplicate rows based on the card_column."""
        duplicate_count = table[card_column].duplicated().sum()
        return duplicate_count

    def check_non_numerical_card_numbers(self, table, card_column):
        """Returns the number of non-numerical card numbers."""
        non_numerical_count = table[card_column].apply(lambda x: not str(x).isdigit()).sum()
        return non_numerical_count

    def remove_null(self, table):
        """Replaces 'NULL' with NaN and removes rows with missing values."""
        table = pd.DataFrame(table)  # Ensure table is a DataFrame
        null_values = ["NULL"]
        table.replace(null_values, np.nan, inplace=True)  # Replace "NULL" strings with NaN
        table.dropna(inplace=True)  # Drop rows with NaN values
        table.reset_index(drop=True, inplace=True)  # Reset index after dropping rows
        return table

    def remove_duplicate_card_numbers(self, table, card_column):
        """Removes duplicate card numbers."""
        return table.drop_duplicates(subset=[card_column])

    def remove_non_numerical_card_numbers(self, table, card_column):
        """Removes rows with non-numerical card numbers."""
        table = table[table[card_column].apply(lambda x: str(x).isdigit())]
        return table

    def convert_to_datetime(self, table, column_name):
        """Converts a column to datetime format."""
        if column_name in table.columns:
            table[column_name] = pd.to_datetime(table[column_name], errors='coerce')  # Convert to datetime
        return table

    def clean_card_data(self, table):
        """
        Cleans card data by:
        - Replacing 'NULL' strings with NaN
        - Dropping rows with NaN values in 'date_payment_confirmed'
        - Removing duplicate card numbers
        - Removing non-numerical card numbers
        - Stripping unwanted characters from 'card_number'
        - Converting 'expiry_date' column to string and filtering valid expiry dates
        - Converting 'date_payment_confirmed' column to datetime format
        """
        clean_card_data = pd.DataFrame(table)  # Ensure table is a DataFrame
        
        # Replaces 'NULL' with NaN and drops specific NaN rows
        clean_card_data = clean_card_data.replace("NULL", np.nan)
        clean_card_data = clean_card_data.dropna(subset=["date_payment_confirmed"])  # Drop rows with NaN in 'date_payment_confirmed'
        
        # Filters out incorrect expiry dates (valid expiry date should be length 5, e.g., 'MM/YY')
        clean_card_data["expiry_date"] = clean_card_data["expiry_date"].astype("string")
        clean_card_data = clean_card_data.query("expiry_date.str.len() == 5")
        
        # Removes unwanted characters from 'card_number' column (e.g., '?')
        clean_card_data["card_number"] = clean_card_data["card_number"].astype("string")
        clean_card_data["card_number"] = clean_card_data["card_number"].str.replace("?", "")  # Removes '?' characters
        
        # Remove 'NULL' strings and drop rows with NaN values
        clean_card_data = self.remove_null(clean_card_data)  # Removing rows with NaN values after replacing 'NULL'
        
        # Remove duplicate card numbers
        clean_card_data = self.remove_duplicate_card_numbers(clean_card_data, 'card_number')  # Remove duplicates
        
        # Remove non-numerical card numbers (remove rows with non-numeric characters)
        clean_card_data = self.remove_non_numerical_card_numbers(clean_card_data, 'card_number')  # Remove non-numerical card numbers
        
        # Convert 'date_payment_confirmed' to datetime
        clean_card_data = self.convert_to_datetime(clean_card_data, 'date_payment_confirmed')  # Convert to datetime
        
        clean_card_data.reset_index(drop=True, inplace=True)  # Reset index after cleaning
        return clean_card_data
```

### Store Details

#### DataExtractor Class 
```
class DataExtractor:
    def __init__(self):
        self.headers = {"x-api-key": "yFBQbwXe9J3sd6zWVAMrK6lcxxr0q1lr2PT6DDMX"}

    def list_number_of_stores(self, number_stores_endpoint, headers):
        """
        Makes a GET request to the number of stores endpoint and returns the number of stores.
        """
        response = requests.get(number_stores_endpoint, headers=headers)
        
        if response.status_code == 200:
            print("API Response:", response.json())  # Debug: print response to check the structure
            response_data = response.json()
            if "number_stores" in response_data and isinstance(response_data["number_stores"], int):
                return response_data["number_stores"]
            else:
                raise ValueError("The API response does not contain a valid 'number_stores' value.")
        else:
            raise Exception(f"Failed to retrieve data: {response.status_code}, {response.text}")

    def retrieve_stores_data(self, retrieve_store_endpoint, headers, num_stores):
        """
        Retrieves store data from the API for each store and saves it in a pandas DataFrame.
        """
        store_data = []

        for store_number in range(0, num_stores):
            store_url = retrieve_store_endpoint.format(store_number=store_number)
            response = requests.get(store_url, headers=headers)
            
            if response.status_code == 200:
                store_data.append(response.json())  # Assuming JSON response
            else:
                print(f"Failed to retrieve data for store {store_number}: {response.status_code}, {response.text}")

            time.sleep(0.2)
        
        df = pd.DataFrame(store_data)
        return df
```

#### DataCleaning Class 
```
class DataCleaning:
    def __init__(self):
        pass

    def clean_store_data(self, store_data_df):
        """
        Cleans the store data DataFrame by performing the following:
        - Replacing "NULL" string with NaN
        - Removing rows with NaN values
        - Filtering country codes (GB, DE, US)
        - Replacing incorrect region names
        - Formatting "opening_date" correctly
        - Cleaning "staff_numbers" by removing letters
        """

        # Step 1: Set index value if "index" column exists
        if "index" in store_data_df.columns:
            store_data_df.set_index("index", inplace=True)

        # Step 2: Filter valid country codes
        valid_countries = ["GB", "DE", "US"]
        if "country_code" in store_data_df.columns:
            store_data_df = store_data_df[store_data_df["country_code"].isin(valid_countries)]

        # Step 3: Replace incorrect continent names using .loc[]
        store_data_df.loc[store_data_df["continent"] == "eeEurope", "continent"] = "Europe"
        store_data_df.loc[store_data_df["continent"] == "eeAmerica", "continent"] = "America"

        # Step 4: Drop "lat" column if it exists
        if "lat" in store_data_df.columns:
            store_data_df.drop(columns=["lat"], inplace=True)

        # Step 5: Replace newlines in address column if it exists
        if "address" in store_data_df.columns:
            store_data_df["address"] = store_data_df["address"].str.replace("\n", " ", regex=True)

        # Step 6: Convert "opening_date" column into a valid datetime format using custom parser
        def custom_date_parser(date_str):
            try:
                return parser.parse(date_str, dayfirst=False)  # Automatically detects format
            except Exception:
                return pd.NaT  # Assign NaT if parsing fails

        if "opening_date" in store_data_df.columns:
            store_data_df["opening_date"] = store_data_df["opening_date"].apply(custom_date_parser)

        # Step 7: Remove letters from "staff_numbers" using .loc[]
        if "staff_numbers" in store_data_df.columns:
            store_data_df["staff_numbers"] = store_data_df["staff_numbers"].str.replace("\\D", "", regex=True)

        # Step 8: Convert empty strings in "staff_numbers" to NaN
        store_data_df["staff_numbers"].replace("", np.nan, inplace=True)

        return store_data_df
```
### Product Details

#### DataExtractor Class
```
import boto3
import pandas as pd
from io import StringIO

class DataExtractor:
    def extract_from_s3(self, s3_path):
        """
        Extracts product CSV from an AWS S3 bucket using S3 address to return a dataframe.
        """
        try:
            # Reads CSV directly from S3
            s3_client = boto3.client('s3')
            bucket_name, file_key = s3_path.replace("s3://", "").split("/", 1)
            obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
            data = obj['Body'].read().decode('utf-8')
            products_data = pd.read_csv(StringIO(data), index_col=0)
            return products_data
        except Exception as e:
            print(f"Error extracting data from S3: {e}")
            return None
```

#### DataCleaning Class 
```
import pandas as pd
import numpy as np

class DataCleaning:
    @staticmethod
    def clean_weights(weight):
        """
        Cleans all weights to the same units (kg) in products data.
        """
        try:
            if "kg" in weight:
                weight = float(weight.replace("kg", ""))
            elif "x" in weight:
                weight = weight.replace("g", "")
                weight_list = weight.split(" x ")
                weight_list = [float(i) for i in weight_list]
                weight = weight_list[0] * weight_list[1] / 1000
            elif "g ." in weight:
                weight = float(weight.replace("g .", "")) / 1000
            elif "g" in weight:
                weight = float(weight.replace("g", "")) / 1000
            elif "ml" in weight:
                weight = float(weight.replace("ml", "")) / 1000
            elif "oz" in weight:
                weight = float(weight.replace("oz", "")) * 0.0283495231
            else:
                return np.nan  # If the weight format is unknown, return NaN
            return round(weight, 3)
        except Exception as e:
            print(f"Error cleaning weight: {e}")
            return np.nan

    def convert_product_weights(self, products_data):
        """
        Cleans weights column in products data and returns dataframe. 
        """
        try:
            clean_products_data = products_data.replace("NULL", np.nan).dropna()
            removed = ["Still_avaliable", "Removed"]
            mask = clean_products_data["removed"].isin(removed)
            clean_products_data = clean_products_data[mask]

            clean_products_data = clean_products_data.replace("Still_avaliable", "Still available")
            clean_products_data["weight"] = clean_products_data["weight"].astype("string")
            clean_products_data["weight"] = clean_products_data["weight"].apply(self.clean_weights)
            return clean_products_data
        except Exception as e:
            print(f"Error converting product weights: {e}")
            return None

    def clean_products_data(self, products_data):
        """
        Cleans products data and returns dataframe. 
        """
        try:
            clean_products_data = self.convert_product_weights(products_data)

            # Cleaning other columns
            clean_products_data["product_price"] = clean_products_data["product_price"].str.replace("£", "").astype(float)
            clean_products_data["category"] = clean_products_data["category"].str.replace("-", " ")
            clean_products_data["date_added"] = pd.to_datetime(clean_products_data["date_added"], errors='coerce')

            return clean_products_data
        except Exception as e:
            print(f"Error cleaning products data: {e}")
            return None

```

### Order tables

#### DataExtractor Class
```
import pandas as pd
from database_utils import DatabaseConnector

class DataExtractor:
    def __init__(self):
        self.db_connector = DatabaseConnector()  # Initialize DatabaseConnector instance

    def extract_from_db(self, query: str) -> pd.DataFrame:
        """
        Extract data from the RDS database using a query and return the result as a DataFrame.
        """
        engine = self.db_connector.init_db_engine()
        return pd.read_sql(query, engine) if engine else pd.DataFrame()

    def read_rds_table(self, table_name: str) -> pd.DataFrame:
        """
        Reads a table from the RDS database and returns it as a Pandas DataFrame.
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
```

#### DataCleaning Class 
```
import pandas as pd
import numpy as np

class DataCleaning:
    def clean_orders_data(self, table):
        """
        Cleans orders data by:
        - Removing unnecessary columns: first_name, last_name, and '1'
        - Ensuring the table is in the correct format for upload to the database
        - This table will act as the source of truth for your sales data.
        """
        # Ensure table is a DataFrame
        table = pd.DataFrame(table)

        # Drop unnecessary columns: 'first_name', 'last_name', and '1'
        columns_to_drop = ['first_name', 'last_name', '1','level_0']
        table.drop(columns=[col for col in columns_to_drop if col in table.columns], inplace=True)

        # Reset the index after removing columns
        table.reset_index(drop=True, inplace=True)

        # You can perform other cleaning actions (like handling duplicates or outliers here if needed)

        print("Cleaned orders data:")
        # print(table.head())  # Print the first few rows to verify

        return table
```

### Events Data

#### DataExtractor Class
```
import pandas as pd
import requests

class DataExtractor:
    def extract_s3_datetime(self, url):
        try:
            # Fetch data from the public S3 URL
            response = requests.get(url)
            response.raise_for_status()  # Raise an error for bad responses
            
            # Convert JSON to a pandas DataFrame
            df = pd.DataFrame(response.json())
            
            return df

        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            return None

```

#### DataCleaning Class 
```
import pandas as pd
import numpy as np

class DataCleaner:
    def clean_datetime_data(self, df):
        """Cleans extracted datetime data."""
        if df is None or df.empty:
            print("DataFrame is empty or None. Skipping cleaning.")
            return df

        # Replace 'NULL' strings with NaN
        df.replace('NULL', np.nan, inplace=True)

        # Drop rows with NaN values
        df.dropna(inplace=True)

        # Convert 'day', 'month', and 'year' to numeric (invalid values become NaN)
        df['day'] = pd.to_numeric(df['day'], errors='coerce')
        df['month'] = pd.to_numeric(df['month'], errors='coerce')
        df['year'] = pd.to_numeric(df['year'], errors='coerce')

        # Drop any new NaN values created after conversion
        df.dropna(inplace=True)

        # Convert to integer type (avoid errors with NaN by using Int64)
        df['day'] = df['day'].astype('Int64')
        df['month'] = df['month'].astype('Int64')
        df['year'] = df['year'].astype('Int64')

        return df

```

## Milestone 3



## Milestone 4

