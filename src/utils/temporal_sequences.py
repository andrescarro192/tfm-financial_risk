# src/utils/temporal_sequences.py
"""
Construcción de secuencias temporales y Dataset PyTorch para el modelo
supervisado (LSTM Baseline e Híbrido). Reutilizable sin cambios entre
ambos: la única entrada que varía es el DataFrame de embeddings (192
columnas para Baseline, 256 para Híbrido) y la lista cols_emb
correspondiente.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def asignar_period_idx(df: pd.DataFrame, period_col: str = "period"):
    """
    Asigna un índice entero consecutivo a cada periodo único (orden
    cronológico), necesario para detectar huecos en
    build_sequences_supervised. Ordena por CERT y period_idx.
    """
    all_periods_sorted = sorted(df[period_col].unique())
    period_to_idx = {p: i for i, p in enumerate(all_periods_sorted)}

    df = df.copy()
    df["period_idx"] = df[period_col].map(period_to_idx)
    df = df.sort_values(["CERT", "period_idx"]).reset_index(drop=True)

    return df, period_to_idx, all_periods_sorted


def calcular_frontera_train_val(all_periods_sorted: list, n_val_periods: int, verbose: bool = True):
    """
    Frontera por número de periodos finales reservados a val, no por
    fracción ni fecha fija — consistente con el diagnóstico empírico
    que fijó N_VAL_PERIODS=8.
    """
    val_periods = set(all_periods_sorted[-n_val_periods:])
    train_periods = set(all_periods_sorted[:-n_val_periods])

    if verbose:
        print(f"Frontera train/val (N_VAL_PERIODS={n_val_periods}):")
        print(f"  Train period_end hasta : {max(train_periods)}")
        print(f"  Val period_end desde   : {min(val_periods)}")
        print(f"  Periodos en val        : {sorted(val_periods)}")

    return train_periods, val_periods


def build_sequences_supervised(
    df: pd.DataFrame,
    cols_emb: list[str],
    window_len: int,
    stride: int,
    val_periods: set,
    verbose: bool = True,
):
    """
    Ventanas deslizantes por CERT, con 'failed' en period_end como
    target supervisado. Requiere que df ya tenga 'period_idx' (ver
    asignar_period_idx).

    El parámetro period_to_idx de la versión original nunca se usaba
    dentro del cuerpo de la función; se elimina aquí, sin cambio de
    comportamiento.
    """
    sequences_train, sequences_val = [], []
    n_skipped_gaps = 0

    for cert, group in df.groupby("CERT", sort=False):
        group = group.sort_values("period_idx").reset_index(drop=True)
        n_obs = len(group)
        if n_obs < window_len:
            continue

        X_banco = group[cols_emb].values
        periods = group["period"].values
        period_idxs = group["period_idx"].values
        failed_vals = group["failed"].values

        for start in range(0, n_obs - window_len + 1, stride):
            end = start + window_len

            if not np.all(np.diff(period_idxs[start:end]) == 1):
                n_skipped_gaps += 1
                continue

            period_end = periods[end - 1]
            target = int(failed_vals[end - 1])

            seq = {
                "X": X_banco[start:end], "CERT": cert,
                "period_end": period_end, "failed": target,
            }

            if period_end in val_periods:
                sequences_val.append(seq)
            else:
                sequences_train.append(seq)

    if verbose:
        print(f"\nVentanas descartadas por huecos: {n_skipped_gaps}")

    return sequences_train, sequences_val


def verificar_secuencias_supervisadas(
    sequences_train: list, sequences_val: list, val_periods: set,
    window_len: int, cols_emb: list[str], verbose: bool = True,
):
    """
    Mismas verificaciones ya establecidas para el Baseline, reutilizadas
    sin cambios para el Híbrido: separación temporal estricta, shape
    esperado, y presencia de positivos en val (si falla, la frontera no
    se respeta igual que en el Baseline — señal de revisar los datos de
    entrada del Híbrido, no del propio código).
    """
    n_train, n_val = len(sequences_train), len(sequences_val)
    n_pos_train = sum(s["failed"] for s in sequences_train)
    n_pos_val = sum(s["failed"] for s in sequences_val)

    assert {s["period_end"] for s in sequences_train}.isdisjoint(val_periods), (
        "Contaminación temporal: period_ends de train solapan con val_periods"
    )
    assert sequences_train[0]["X"].shape == (window_len, len(cols_emb)), (
        f"Shape esperado ({window_len}, {len(cols_emb)}), "
        f"obtenido {sequences_train[0]['X'].shape}"
    )
    assert n_pos_val > 0, "Val no contiene positivos pese a la frontera fijada."

    if verbose:
        print("\n" + "=" * 60)
        print("BLOQUE — VERIFICACIÓN DE SECUENCIAS (target supervisado)")
        print("=" * 60)
        print(f"Secuencias train         : {n_train:,}")
        print(f"  Positivos en train     : {n_pos_train}  (tasa: {n_pos_train/n_train:.6f})")
        print(f"Secuencias val           : {n_val:,}")
        print(f"  Positivos en val       : {n_pos_val}  (tasa: {n_pos_val/n_val:.6f})")
        print(f"Total secuencias         : {n_train + n_val:,}")
        print(f"Shape por secuencia      : {sequences_train[0]['X'].shape}")
        print("=" * 60)

    return {"n_train": n_train, "n_val": n_val, "n_pos_train": n_pos_train, "n_pos_val": n_pos_val}

# Funciónes para la creación de secuencias de evaluación, sin la separación train/val

def build_sequences_evaluation(
    df: pd.DataFrame,
    cols_emb: list[str],
    window_len: int,
    stride: int,
    verbose: bool = True,
):
    """
    Ventanas deslizantes por CERT, con 'failed' en period_end como
    target supervisado. Versión para bloque de evaluación: sin reparto
    train/val, todas las secuencias generadas pertenecen al mismo
    conjunto. Requiere que df ya tenga 'period_idx' (ver
    asignar_period_idx).

    Misma lógica de detección de huecos que build_sequences_supervised
    (np.diff(period_idxs) == 1), porque la integridad temporal de la
    ventana no depende de si hay partición train/val o no: una ventana
    con un hueco interno es inválida en cualquier bloque.
    """
    sequences_eval = []
    n_skipped_gaps = 0

    for cert, group in df.groupby("CERT", sort=False):
        group = group.sort_values("period_idx").reset_index(drop=True)
        n_obs = len(group)
        if n_obs < window_len:
            continue

        X_banco = group[cols_emb].values
        periods = group["period"].values
        period_idxs = group["period_idx"].values
        failed_vals = group["failed"].values

        for start in range(0, n_obs - window_len + 1, stride):
            end = start + window_len

            if not np.all(np.diff(period_idxs[start:end]) == 1):
                n_skipped_gaps += 1
                continue

            period_end = periods[end - 1]
            target = int(failed_vals[end - 1])

            sequences_eval.append({
                "X": X_banco[start:end], "CERT": cert,
                "period_end": period_end, "failed": target,
            })

    if verbose:
        print(f"\nVentanas descartadas por huecos: {n_skipped_gaps}")

    return sequences_eval


def verificar_secuencias_evaluacion(
    sequences_eval: list, window_len: int, cols_emb: list[str], verbose: bool = True,
):
    """
    Verificaciones equivalentes a verificar_secuencias_supervisadas, sin
    la comprobación de disjunción train/val (no aplica: no hay
    partición en este bloque). Se mantienen las comprobaciones de shape
    y presencia de positivos, por la misma razón que en train/val: un
    fallo aquí indica problema en los datos de entrada, no en el código
    de ventaneo.
    """
    n_eval = len(sequences_eval)
    n_pos_eval = sum(s["failed"] for s in sequences_eval)

    assert n_eval > 0, "No se generó ninguna secuencia de evaluación."
    assert sequences_eval[0]["X"].shape == (window_len, len(cols_emb)), (
        f"Shape esperado ({window_len}, {len(cols_emb)}), "
        f"obtenido {sequences_eval[0]['X'].shape}"
    )
    assert n_pos_eval > 0, "Evaluación no contiene positivos."

    if verbose:
        print("\n" + "=" * 60)
        print("BLOQUE — VERIFICACIÓN DE SECUENCIAS DE EVALUACIÓN")
        print("=" * 60)
        print(f"Secuencias evaluación    : {n_eval:,}")
        print(f"  Positivos en evaluación: {n_pos_eval}  (tasa: {n_pos_eval/n_eval:.6f})")
        print(f"Shape por secuencia      : {sequences_eval[0]['X'].shape}")
        print("=" * 60)

    return {"n_eval": n_eval, "n_pos_eval": n_pos_eval}


def calcular_pos_weight_raw(sequences_train: list, verbose: bool = True) -> float:
    """pos_weight exclusivamente sobre train, nunca val ni evaluación."""
    n_train = len(sequences_train)
    n_pos_train = sum(s["failed"] for s in sequences_train)
    pos_weight_raw = (n_train - n_pos_train) / n_pos_train

    if verbose:
        print(f"\npos_weight (ratio bruto neg/pos en train): {pos_weight_raw:.2f}")

    return pos_weight_raw


class SupervisedSequenceDataset(Dataset):
    """CERT y period_end como metadato: trazabilidad, evaluación por
    entidad, y el análisis de ranking/top-K por trimestre."""

    def __init__(self, sequences: list) -> None:
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        seq = self.sequences[idx]
        return {
            "X": torch.tensor(seq["X"], dtype=torch.float32),
            "failed": torch.tensor(seq["failed"], dtype=torch.float32),
            "CERT": seq["CERT"],
            "period_end": seq["period_end"],
        }