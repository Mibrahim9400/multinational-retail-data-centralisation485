import pandas as pd
import numpy as np
from dateutil import parser
from typing import Optional

class DataCleaning:
    def __init__(self):
        pass

    def _replace_null_with_nan(self, table: pd.DataFrame) -> pd.DataFrame:
        """Replaces 'NULL' strings with NaN and removes rows with NaN values."""
        table.replace("NULL", np.nan, inplace=True)  # Replace "NULL" with NaN
        table.dropna(inplace=True)  # Drop rows with NaN values
        table.reset_index(drop=True, inplace=True)  # Reset index after dropping rows
        return table

    def _filter_valid_date(self, table: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """Cleans and converts a date column to datetime, ensuring only valid dates remain."""
        if date_column in table.columns:
            non_date_mask = table[date_column].astype(str).str.match(r'^[A-Za-z0-9]+$', na=False)
            table = table[~non_date_mask]
            table[date_column] = pd.to_datetime(table[date_column], format="mixed", errors="coerce")
        return table
    
    def clean_user_data(self, table: pd.DataFrame) -> pd.DataFrame:
        """Cleans user data by removing 'NULL', dropping NaNs, and ensuring valid join_date."""
        table = self._replace_null_with_nan(table)  # Remove NULL values and NaNs
        table = self._filter_valid_date(table, 'join_date')  # Clean and convert 'join_date'
        return table

    def check_null_values(self, table: pd.DataFrame) -> pd.Series:
        """Returns the number of NULL values in each column."""
        return table.isnull().sum()

    def check_duplicate_rows(self, table: pd.DataFrame, card_column: str) -> int:
        """Returns the number of duplicate rows based on the specified card column."""
        return table[card_column].duplicated().sum()

    def check_non_numerical_card_numbers(self, table: pd.DataFrame, card_column: str) -> int:
        """Returns the number of non-numerical card numbers."""
        return table[card_column].apply(lambda x: not str(x).isdigit()).sum()

    def remove_duplicate_card_numbers(self, table: pd.DataFrame, card_column: str) -> pd.DataFrame:
        """Removes duplicate card numbers."""
        return table.drop_duplicates(subset=[card_column])

    def remove_non_numerical_card_numbers(self, table: pd.DataFrame, card_column: str) -> pd.DataFrame:
        """Removes rows with non-numerical card numbers."""
        return table[table[card_column].apply(lambda x: str(x).isdigit())]

    def convert_to_datetime(self, table: pd.DataFrame, column_name: str) -> pd.DataFrame:
        """Converts a column to datetime format."""
        if column_name in table.columns:
            table[column_name] = pd.to_datetime(table[column_name], errors='coerce')
        return table

    def clean_card_data(self, table: pd.DataFrame) -> pd.DataFrame:
        """Cleans card data by removing duplicates, non-numerical card numbers, and invalid expiry dates."""
        clean_card_data = self._replace_null_with_nan(table)  # Remove NULL values and drop NaNs
        clean_card_data = self.remove_duplicate_card_numbers(clean_card_data, 'card_number')
        clean_card_data = self.remove_non_numerical_card_numbers(clean_card_data, 'card_number')
        clean_card_data = self.convert_to_datetime(clean_card_data, 'date_payment_confirmed')
        
        # Clean expiry_date and remove unwanted characters in card_number
        clean_card_data["expiry_date"] = clean_card_data["expiry_date"].astype("string")
        clean_card_data = clean_card_data.query("expiry_date.str.len() == 5")  # Valid expiry length 'MM/YY'
        clean_card_data["card_number"] = clean_card_data["card_number"].str.replace("?", "")  # Clean card_number

        clean_card_data.reset_index(drop=True, inplace=True)  # Reset index after cleaning
        return clean_card_data

    def clean_store_data(self, store_data_df: pd.DataFrame) -> pd.DataFrame:
        """Cleans store data by filtering countries, correcting continent names, and formatting columns."""
        if "index" in store_data_df.columns:
            store_data_df.set_index("index", inplace=True)

        store_data_df = store_data_df[store_data_df["country_code"].isin(["GB", "DE", "US"])]  # Filter valid countries

        # Correct continent names
        store_data_df.loc[store_data_df["continent"] == "eeEurope", "continent"] = "Europe"
        store_data_df.loc[store_data_df["continent"] == "eeAmerica", "continent"] = "America"

        store_data_df.drop(columns=["lat"], inplace=True, errors="ignore")  # Remove 'lat' column if exists
        store_data_df["address"] = store_data_df["address"].str.replace("\n", " ", regex=True)  # Clean address

        # Convert opening_date using a custom parser
        store_data_df["opening_date"] = store_data_df["opening_date"].apply(self._custom_date_parser)

        store_data_df["staff_numbers"] = store_data_df["staff_numbers"].str.replace("\\D", "", regex=True)  # Clean staff_numbers
        store_data_df["staff_numbers"].replace("", np.nan, inplace=True)  # Replace empty strings with NaN

        return store_data_df

    @staticmethod
    def _custom_date_parser(date_str: str) -> Optional[pd.Timestamp]:
        """Parses a date string into a datetime object, returning NaT if parsing fails."""
        try:
            return parser.parse(date_str, dayfirst=False)
        except Exception:
            return pd.NaT  # Return NaT if parsing fails

    def clean_weights(self, weight: str) -> Optional[float]:
        """Cleans weights and converts them to kg units."""
        try:
            if "kg" in weight:
                return float(weight.replace("kg", ""))
            elif "x" in weight:
                weight_list = weight.replace("g", "").split(" x ")
                return float(weight_list[0]) * float(weight_list[1]) / 1000
            elif "g" in weight:
                return float(weight.replace("g", "")) / 1000
            elif "ml" in weight:
                return float(weight.replace("ml", "")) / 1000
            elif "oz" in weight:
                return float(weight.replace("oz", "")) * 0.0283495231
            return np.nan
        except Exception as e:
            print(f"Error cleaning weight: {e}")
            return np.nan

    def convert_product_weights(self, products_data: pd.DataFrame) -> pd.DataFrame:
        """Cleans weights column and returns cleaned dataframe."""
        try:
            products_data = products_data.replace("NULL", np.nan).dropna()
            products_data = products_data[products_data["removed"].isin(["Still available"])]  # Filter removed products
            products_data["weight"] = products_data["weight"].apply(self.clean_weights)
            return products_data
        except Exception as e:
            print(f"Error converting product weights: {e}")
            return pd.DataFrame()

    def clean_products_data(self, products_data: pd.DataFrame) -> pd.DataFrame:
        """Cleans product data by converting weights, formatting prices, and ensuring correct datetime formats."""
        try:
            products_data = self.convert_product_weights(products_data)
            products_data["product_price"] = products_data["product_price"].str.replace("£", "").astype(float)
            products_data["category"] = products_data["category"].str.replace("-", " ")
            products_data["date_added"] = pd.to_datetime(products_data["date_added"], errors='coerce')
            return products_data
        except Exception as e:
            print(f"Error cleaning products data: {e}")
            return pd.DataFrame()

    def clean_orders_data(self, table: pd.DataFrame) -> pd.DataFrame:
        """Cleans orders data by removing unnecessary columns and preparing for database upload."""
        columns_to_drop = ['first_name', 'last_name', '1', 'level_0']
        table.drop(columns=[col for col in columns_to_drop if col in table.columns], inplace=True)
        table.reset_index(drop=True, inplace=True)
        return table

    def clean_datetime_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans extracted datetime data by converting to numeric and handling errors."""
        if df is None or df.empty:
            return df

        df.replace('NULL', np.nan, inplace=True)
        df.dropna(inplace=True)
        df['day'] = pd.to_numeric(df['day'], errors='coerce')
        df['month'] = pd.to_numeric(df['month'], errors='coerce')
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df.dropna(inplace=True)
        df['day'] = df['day'].astype('Int64')
        df['month'] = df['month'].astype('Int64')
        df['year'] = df['year'].astype('Int64')
        return df