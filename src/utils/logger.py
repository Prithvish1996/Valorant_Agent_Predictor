import logging
from pathlib import Path


class Logger:

    _initialized = False
    _log_dir = Path(__file__).resolve().parent.parent.parent / "logs"

    @classmethod
    def setup(cls, name: str, level: int = logging.INFO) -> logging.Logger:
        # Create logs directory if it doesn't exist
        cls._log_dir.mkdir(exist_ok=True)

        logger = logging.getLogger(name)
        logger.setLevel(level)

        if not logger.handlers:
            file_handler = logging.FileHandler(cls._log_dir / "app.log")
            file_handler.setLevel(level)


            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        return logger
