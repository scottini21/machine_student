import unittest
from unittest.mock import MagicMock, patch
from pyspark.sql import SparkSession, DataFrame
import pandas as pd
from machine_student.feature_pipeline.retrieval import DataRetriever  # Adjust import based on your project structure

class TestDataRetriever(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Set up Spark session for testing."""
        cls.spark = SparkSession.builder \
            .master("local[*]") \
            .appName("test") \
            .getOrCreate()

    @classmethod
    def tearDownClass(cls):
        """Stop the Spark session after tests."""
        cls.spark.stop()

    def setUp(self):
        """Set up sample data for each test."""
        self.data = [("1", "Alice", 25), ("2", "Bob", 30), ("3", "Charlie", 35)]
        self.columns = ["id", "name", "age"]
        self.sample_data = self.spark.createDataFrame(self.data, self.columns)
        self.file_path = "dummy_path"
        self.file_format = "csv"
        self.id_column = "id"

    def test_get_spark_session(self):
        retriever = DataRetriever(
            file_path=self.file_path,
            file_format=self.file_format,
            id_column=self.id_column
        )
        self.assertIsInstance(retriever.spark, SparkSession)

    @patch('machine_student.data.retrieval.SparkSession.read')
    def test_get_df_reader(self, mock_read):
        # Mock DataFrameReader
        mock_reader = MagicMock()
        mock_read.return_value = mock_reader
        
        retriever = DataRetriever(
            file_path=self.file_path,
            file_format=self.file_format,
            id_column=self.id_column
        )
        
        reader = retriever.get_df_reader()
        
        # Verify that format() was called with the correct file format
        #mock_reader.format.assert_called_with(self.file_format)
        
        # Ensure the returned reader is the mocked reader
        self.assertEqual(reader, mock_reader)

    @patch('pyspark.sql.SparkSession.read')
    def test_get_raw_data(self, mock_read):
        # Mock DataFrameReader and its methods
        mock_reader = MagicMock()
        mock_read.return_value = mock_reader
        
        # Set up the mock to return the sample_data DataFrame when load() is called
        mock_reader.format.return_value = mock_reader
        mock_reader.load.return_value = self.sample_data
        
        retriever = DataRetriever(
            file_path=self.file_path,
            file_format=self.file_format,
            id_column=self.id_column
        )
        
        df_raw = retriever.get_raw_data()
        
        # Verify that load() was called with the correct file path
        mock_reader.load.assert_called_with(self.file_path)
        
        # Verify that the returned DataFrame is the sample_data DataFrame
        self.assertEqual(df_raw.collect(), self.sample_data.collect())

    def test_drop_columns(self):
        retriever = DataRetriever(
            file_path=self.file_path,
            file_format=self.file_format,
            id_column=self.id_column,
            columns_to_drop=["name"]
        )
        df_dropped = retriever.drop_columns(self.sample_data)
        self.assertNotIn("name", df_dropped.columns)

    def test_transform_to_pandas(self):
        retriever = DataRetriever(
            file_path=self.file_path,
            file_format=self.file_format,
            id_column=self.id_column
        )
        df_pandas = retriever.transform_to_pandas(self.sample_data)
        self.assertIsInstance(df_pandas, pd.DataFrame)
        self.assertEqual(df_pandas.index.name, self.id_column)

    def test_execute(self):
        retriever = DataRetriever(
            file_path=self.file_path,
            file_format=self.file_format,
            id_column=self.id_column,
            columns_to_drop=["name"]
        )
        retriever.get_raw_data = MagicMock(return_value=self.sample_data)
        df_pandas = retriever.execute()
        self.assertIsInstance(df_pandas, pd.DataFrame)
        self.assertEqual(df_pandas.index.name, self.id_column)
        self.assertNotIn("name", df_pandas.columns)


if __name__ == "__main__":
    unittest.main()
