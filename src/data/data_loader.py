import pandas as pd
from pathlib import Path

from src.utils.logger import Logger

logger = Logger.setup(__name__)


class DataLoader:
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir).resolve()
        logger.info(f"DataLoader initialized with data directory: {self.data_dir}")

    def load_data(self, filename: str) -> pd.DataFrame:
        file_path = self.data_dir / filename
        logger.info(f"Attempting to load file: {file_path}")

        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File {file_path} does not exist.")

        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded {filename} with {len(df)} rows and {len(df.columns)} columns")
        return df
