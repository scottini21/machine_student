from typing import List, Dict, Any, Optional

import pandas as pd
from pyspark.sql import SparkSession, DataFrame, DataFrameReader


class DataRetriever:
    """
    DataRetriever class it is used to get the data from a certain file
    so it can be used as a simple step for the ML process.

    Attributes
    ----------
    file_path : str
        File location of the file that contains the data.
    file_format : str
        File format of the file that contains the data that you want to read.
    id_column : str
        Name of the column that will be used to identify each register.
        It will appear as an index in the final execution.
    read_kwargs : Optional[Dict[str,Any]]= None
        Additional configuration that may be required to read a certain file.
    columns_to_drop : Optional[List[str]]= None
        List containing columns names to drop from raw data.
        If None is provided then it leaves the raw data as it is.
    spark :  SparkSession
        Spark Session that is constructed to retrieve data efficiently.
    """
    def __init__(self,
                 file_path: str,
                 file_format: str,
                 id_column: str,
                 read_kwargs: Optional[Dict[str, Any]] = None,
                 columns_to_drop: Optional[List[str]] = None
                 ):
        """
        This function is the DataRetriever object constructor.

        Parameters
        ----------
        file_path : str
            File location of the file that contains the data.
        file_format : str
            File format of the file that contains the data that you want
            to read.
        id_column : str
            Name of the column that will be used to identify each register
            that will be used in the ML model.
        read_kwargs : Optional[Dict[str,Any]]= None
            Additional configuration that may be required to read
            a certain file.
        columns_to_drop : Optional[List[str]]= None
           List containing columns names to drop from raw data.
           If None is provided then it leaves the raw data as it is.
        """
        self.file_path = file_path
        self.file_format = file_format
        self.id_column = id_column
        self.read_kwargs = read_kwargs
        self.columns_to_drop = columns_to_drop
        self.spark = self.get_spark_session()

    def get_spark_session(self) -> SparkSession:
        """
        get_spark_session .

        Function that creates or gets the Spark session to retrieve the data.

        Returns
        -------
        SparkSession
            With the proper name for this use case.
        """
        spark = (SparkSession.builder
                 .getOrCreate())
        return spark

    def get_df_reader(self) -> DataFrameReader:
        """
        Function that sets up the reading configuration.-

        Returns
        -------
        DataFrameReader
            Object that will fetch the data with the specified configuration.
        """
        reader = (self.spark.read.format(self.file_format))
        if self.read_kwargs:
            reader = reader.options(**self.read_kwargs)
        return reader

    def get_raw_data(self) -> DataFrame:
        """

        Function that retrieves the data as it is in the file.

        Returns
        -------
        DataFrame
            SparkDataFrame contianing all raw data.
        """
        reader = self.get_df_reader()
        df_raw = reader.load(self.file_path)
        return df_raw

    def drop_columns(self, df: DataFrame) -> DataFrame:
        """

        Function that drops the columns specified in the constructor
        from given DataFrame.

        Parameters
        ----------
        df : DataFrame
            DataFrame where the columns needs to be dropped.

        Returns
        -------
        DataFrame
            DataFrame withoud the dropped columns.
        """
        if self.columns_to_drop:
            df = df.drop(*self.columns_to_drop)
        return df

    def transform_to_pandas(self, df: DataFrame) -> pd.DataFrame:
        """

        Function that transforms a spark dataframe into a pandas
        dataframe and sets up the index as the id_column of the
        object.

        Parameters
        ----------
        df : DataFrame
            DataFrame that needs to be transported to pandas.

        Returns
        -------
        pd.DataFrame
            Pandas DataFrame with the id_column from constructor
            as index
        """
        df_pandas = df.toPandas()
        df_pandas = df_pandas.set_index(self.id_column)
        return df_pandas

    def execute(self) -> pd.DataFrame:
        """

        Main function of this class that returns the whole pandas
        dataframe that contains all the data used in model building
        process. It is composed of previous methods in this class.

        Returns
        -------
        pd.DataFrame
            Final pandas DataFrame that contains the data that
            will build the models.
        """
        df_raw = self.get_raw_data()
        df_selection = self.drop_columns(df_raw)
        df_pandas = self.transform_to_pandas(df_selection)
        return df_pandas
