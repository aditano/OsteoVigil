from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(output_dir: Path, level: str = "INFO") -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("osteovigil")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    logfile = logging.FileHandler(output_dir / "pipeline.log")
    logfile.setFormatter(formatter)
    logger.addHandler(logfile)

    return logger

