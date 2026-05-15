from pathlib import Path
import pandas as pd
from src.utils.config import RAW_DIR, FAILURES_FILE, ENTITY_COL


def get_quarters() -> list[Path]:
    """Devuelve carpetas trimestrales ordenadas cronológicamente."""
    return sorted([p for p in RAW_DIR.iterdir() if p.is_dir()])


def load_quarter(quarter_path: Path, file_type: str) -> pd.DataFrame:
    """
    Carga un tipo de archivo de un trimestre concreto.
    file_type: 'FTS' | 'CDI' | 'RAT' | 'MERG' | 'STRU'
    """
    suffix = quarter_path.name[3:]  # 'ris1603' -> '1603'
    filepath = quarter_path / f"{file_type}{suffix}.csv"

    if not filepath.exists():
        raise FileNotFoundError(f"No encontrado: {filepath}")

    df = pd.read_csv(filepath, low_memory=False)
    df[ENTITY_COL] = df[ENTITY_COL].astype(str)  # CERT siempre string
    return df


def load_all_quarters(file_type: str) -> pd.DataFrame:
    """
    Carga y concatena un tipo de archivo de todos los trimestres.
    Añade columna 'period' con el trimestre de origen (e.g. '1603').
    """
    frames = []
    for quarter_path in get_quarters():
        try:
            df = load_quarter(quarter_path, file_type)
            df["period"] = quarter_path.name[3:]  # '1603', '1606', ...
            frames.append(df)
        except FileNotFoundError as e:
            print(f"Warning: {e}")

    return pd.concat(frames, ignore_index=True)


def load_failures() -> pd.DataFrame:
    """Carga el archivo de quiebras bancarias."""
    return pd.read_csv(FAILURES_FILE)

