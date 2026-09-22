"""
MODULE 1 — DATA COLLECTION
Handles loading, verifying, inspecting, and schema detection for the house dataset.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import config
from src.utils import log_substep

class DataLoader:
    """Class to load and inspect raw house dataset."""
    
    def __init__(self, file_path: Path = config.RAW_DATA_PATH):
        self.file_path = Path(file_path)
        self.df: pd.DataFrame = pd.DataFrame()
        self.numerical_cols: List[str] = []
        self.categorical_cols: List[str] = []

    def load_data(self) -> pd.DataFrame:
        """
        Load CSV from file_path with thorough validation.
        Raises FileNotFoundError with friendly message if missing.
        """
        if not self.file_path.exists():
            raise FileNotFoundError(
                f"House dataset not found at '{self.file_path}'.\n"
                f"Please place the CSV file inside the '{config.DATA_DIR}' folder."
            )
        
        try:
            self.df = pd.read_csv(self.file_path)
        except Exception as e:
            raise RuntimeError(f"Error reading dataset CSV: {e}")
            
        if self.df.empty:
            raise ValueError(f"Dataset at '{self.file_path}' is empty.")
            
        self._inspect_schema()
        return self.df

    def _inspect_schema(self):
        """Identify numerical and categorical features dynamically."""
        self.numerical_cols = self.df.select_dtypes(include=["number"]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(exclude=["number"]).columns.tolist()

    def get_summary_dict(self) -> Dict:
        """Return structured summary of the raw dataset."""
        return {
            "rows": len(self.df),
            "columns": len(self.df.columns),
            "column_names": self.df.columns.tolist(),
            "numerical_cols": self.numerical_cols,
            "categorical_cols": self.categorical_cols,
            "missing_values": self.df.isnull().sum().to_dict(),
            "total_missing": int(self.df.isnull().sum().sum()),
            "duplicates": int(self.df.duplicated().sum()),
            "dtypes": {col: str(dtype) for col, dtype in self.df.dtypes.items()}
        }

    def print_summary(self):
        """Display the required console summary."""
        summary = self.get_summary_dict()
        print("\nDataset Loaded Successfully\n")
        print(f"Rows: {summary['rows']}")
        print(f"Columns: {summary['columns']}")
        print(f"Numerical Features: {len(summary['numerical_cols'])}")
        print(f"Categorical Features: {len(summary['categorical_cols'])}")
        log_substep(f"Verified dataset integrity ({summary['rows']} rows, {summary['columns']} columns)")

def load_and_inspect(file_path: Path = config.RAW_DATA_PATH) -> Tuple[pd.DataFrame, Dict]:
    """Convenience functional interface for Module 1."""
    loader = DataLoader(file_path)
    df = loader.load_data()
    loader.print_summary()
    return df, loader.get_summary_dict()
