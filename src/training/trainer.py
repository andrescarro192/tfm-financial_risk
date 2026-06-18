# src/training/trainer.py
"""
Bucle de entrenamiento reutilizable para HybridLSTMAE.

Este módulo centraliza la lógica de entrenamiento para que sea compartida
por tres usos distintos dentro del proyecto:

    1. Búsqueda de hiperparámetros con Optuna (Bloque 4a):
       entrenamientos cortos (pocas épocas) con pruning activo, optimizando
       learning_rate, weight_decay y dropout sobre val_loss (MSE de
       reconstrucción en validación, métrica NO SUPERVISADA).

    2. Entrenamiento final (Bloque 4):
       entrenamiento completo (EPOCHS=200) con los mejores hiperparámetros
       encontrados, sin pruning.

    3. Ablación (Bloque 7):
       18 ejecuciones (3 configs arquitectónicas x 2 variantes de loss x
       3 seeds), reutilizando esta misma función con los hiperparámetros
       de optimización ya fijados por Optuna.

Principio de diseño: completamente no supervisado
---------------------------------------------------
La etiqueta `failed` / `is_anomalous` NUNCA participa en:
    - el cálculo de la loss de entrenamiento,
    - el criterio de early stopping,
    - el scheduler,
    - el objetivo de Optuna.

Se usa EXCLUSIVAMENTE para una métrica informativa por época: la separación
del MSE de reconstrucción entre observaciones normales y anómalas en train.
Esta métrica no influye en ninguna decisión del entrenamiento; sirve solo
para observar si, a medida que el modelo aprende a reconstruir la normalidad,
el error de reconstrucción de los positivos se separa del de los negativos
(señal de que el anomaly score será informativo en el Bloque 5).

Early stopping con restauración del mejor estado
--------------------------------------------------
Al finalizar (por agotar epochs o por early stopping), el modelo devuelto
tiene cargados los pesos de la época con mejor val_loss, no los de la
última época. Esto es importante porque ReduceLROnPlateau y el propio
ruido del entrenamiento pueden hacer que val_loss empeore en las últimas
épocas antes de detenerse.

Hook de pruning para Optuna
-----------------------------
Si se pasa un objeto `trial` (de Optuna), tras cada época se reporta
val_loss mediante `trial.report(val_loss, epoch)` y se consulta
`trial.should_prune()`. Si el trial debe podarse, se lanza
`optuna.TrialPruned()`.

El import de `optuna` es perezoso (solo ocurre si `trial is not None`),
de forma que este módulo no requiere optuna instalado para los usos 2 y 3
(entrenamiento final y ablación), que no pasan ningún trial.
"""

import copy
import time
from typing import Optional, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.anomaly.losses import TemporalWeightedLoss


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _make_eval_criterion(criterion: TemporalWeightedLoss) -> TemporalWeightedLoss:
    """
    Construye una copia de `criterion` con reduction='none', reutilizando
    los mismos pesos temporales y el mismo modo (uniforme/ponderado).

    Se usa para calcular el MSE por muestra (sin reducir sobre el batch),
    necesario para separar positivos de negativos en la métrica informativa
    de cada época. El criterio principal de entrenamiento usa
    reduction='mean' para el backward; este auxiliar no se usa para
    backpropagation.
    """
    if criterion.use_temporal_weighting and criterion.weights_logical is not None:
        weights = criterion.weights_logical.tolist()
    else:
        weights = None

    return TemporalWeightedLoss(
        weights=weights,
        use_temporal_weighting=criterion.use_temporal_weighting,
        reduction="none",
    )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: TemporalWeightedLoss,
    device: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    criterion_eval: Optional[TemporalWeightedLoss] = None,
) -> dict[str, Any]:
    """
    Ejecuta una pasada completa sobre `loader`.

    Si `optimizer` no es None, se asume modo entrenamiento: el modelo se
    pone en train(), se calcula backward y se actualizan los pesos.
    Si `optimizer` es None, se asume modo evaluación: model.eval() y
    torch.no_grad().

    Si `criterion_eval` (reduction='none') se proporciona, además del
    loss agregado se acumulan los MSE por muestra junto con la máscara
    `is_anomalous`, para la métrica informativa de separación.

    Retorna
    -------
    dict con:
        'loss': float, loss promedio (ponderado por nº de muestras) sobre el loader.
        'mse_normales': list[float] o None
        'mse_anomalos': list[float] o None
    """
    is_training = optimizer is not None
    model.train(mode=is_training)

    total_loss = 0.0
    total_samples = 0

    mse_normales: list[float] = []
    mse_anomalos: list[float] = []
    collect_split = criterion_eval is not None

    context = torch.enable_grad() if is_training else torch.no_grad()

    with context:
        for batch in loader:
            x = batch["X"].to(device)
            batch_size = x.size(0)

            e_proj, x_hat = model(x)
            loss = criterion(e_proj, x_hat)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            if collect_split:
                # mse_per_sample: (batch,) con reduction='none'
                mse_per_sample = criterion_eval(e_proj, x_hat)
                is_anom = batch["is_anomalous"].to(device)

                mse_np = mse_per_sample.detach().cpu()
                anom_np = is_anom.detach().cpu()

                mse_normales.extend(mse_np[~anom_np].tolist())
                mse_anomalos.extend(mse_np[anom_np].tolist())

    return {
        "loss": total_loss / total_samples,
        "mse_normales": mse_normales if collect_split else None,
        "mse_anomalos": mse_anomalos if collect_split else None,
    }


# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------

def train_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: TemporalWeightedLoss,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    epochs: int,
    patience_early_stopping: int,
    patience_scheduler: int,
    factor_scheduler: float,
    trial: Optional[Any] = None,
    verbose: bool = True,
    log_every_n_epochs: int = 5,
    compute_split_metric: bool = True,
) -> dict[str, Any]:
    """
    Entrena `model` minimizando `criterion` sobre `train_loader`, evaluando
    en `val_loader`, con scheduler ReduceLROnPlateau y early stopping sobre
    val_loss.

    La etiqueta `is_anomalous` no influye en loss, scheduler ni early
    stopping. Se usa únicamente para la métrica informativa de separación
    MSE normales vs anómalos en TRAIN (no en val, que no tiene positivos).

    Parámetros
    ----------
    model : nn.Module
        Instancia de HybridLSTMAE, ya movida a `device`.
    optimizer : torch.optim.Optimizer
        Optimizador ya construido (lr y weight_decay se fijan al crearlo,
        por eso no son argumentos de esta función).
    criterion : TemporalWeightedLoss
        Loss con reduction='mean', usada para backward.
    train_loader, val_loader : DataLoader
    device : str
    epochs : int
        Número máximo de épocas.
    patience_early_stopping : int
        Épocas sin mejora de val_loss antes de detener el entrenamiento.
    patience_scheduler, factor_scheduler : ReduceLROnPlateau
        Parámetros del scheduler sobre val_loss.
    trial : optuna.Trial | None
        Si se proporciona, se reporta val_loss por época y se comprueba
        pruning. Lanza optuna.TrialPruned() si el trial debe podarse.
    verbose : bool
    log_every_n_epochs : int
    compute_split_metric : bool
        Si True, calcula la separación MSE normales/anómalos en train en
        cada época (coste adicional despreciable: una pasada con
        reduction='none' sobre el mismo batch ya computado).

    Retorna
    -------
    dict con:
        'model': nn.Module — modelo con los pesos de la mejor época (val_loss mínima).
        'best_val_loss': float
        'best_epoch': int
        'history': dict con listas 'train_loss', 'val_loss', 'lr',
                   'mse_normales_mean', 'mse_anomalos_mean' (las dos
                   últimas son None por época si compute_split_metric=False
                   o si train no contiene anómalos en ese batch).
        'stopped_early': bool
    """
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=factor_scheduler,
        patience=patience_scheduler,
    )

    criterion_eval = _make_eval_criterion(criterion) if compute_split_metric else None

    best_val_loss = float("inf")
    best_epoch = -1
    best_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    stopped_early = False

    history: dict[str, list] = {
        "train_loss": [],
        "val_loss": [],
        "lr": [],
        "mse_normales_mean": [],
        "mse_anomalos_mean": [],
    }

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_result = _run_epoch(
            model, train_loader, criterion, device,
            optimizer=optimizer, criterion_eval=criterion_eval,
        )
        val_result = _run_epoch(
            model, val_loader, criterion, device,
            optimizer=None, criterion_eval=None,
        )

        train_loss = train_result["loss"]
        val_loss = val_result["loss"]

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # --- Métrica informativa: separación MSE normales/anómalos en train ---
        mse_norm_mean = None
        mse_anom_mean = None
        if compute_split_metric:
            mse_normales = train_result["mse_normales"]
            mse_anomalos = train_result["mse_anomalos"]
            if len(mse_normales) > 0:
                mse_norm_mean = sum(mse_normales) / len(mse_normales)
            if len(mse_anomalos) > 0:
                mse_anom_mean = sum(mse_anomalos) / len(mse_anomalos)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)
        history["mse_normales_mean"].append(mse_norm_mean)
        history["mse_anomalos_mean"].append(mse_anom_mean)

        # --- Early stopping sobre val_loss ---
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # --- Logging ---
        if verbose and (epoch % log_every_n_epochs == 0 or epoch == 1 or improved):
            elapsed = time.time() - t0
            split_str = ""
            if mse_norm_mean is not None and mse_anom_mean is not None:
                split_str = (
                    f" | mse_norm={mse_norm_mean:.2e} mse_anom={mse_anom_mean:.2e} "
                    f"ratio={mse_anom_mean / mse_norm_mean:.2f}x"
                )
            marker = " *" if improved else ""
            print(
                f"[Epoch {epoch:3d}/{epochs}] "
                f"train_loss={train_loss:.2e} val_loss={val_loss:.6f} "
                f"lr={current_lr:.2e}{split_str} "
                f"({elapsed:.1f}s){marker}"
            )

        # --- Hook de pruning para Optuna ---
        if trial is not None:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()

        # --- Early stopping ---
        if epochs_no_improve >= patience_early_stopping:
            stopped_early = True
            if verbose:
                print(
                    f"[EarlyStopping] Sin mejora en val_loss durante "
                    f"{patience_early_stopping} épocas. Deteniendo en epoch {epoch}. "
                    f"Mejor val_loss={best_val_loss:.6f} en epoch {best_epoch}."
                )
            break

    # Restaurar los pesos de la mejor época
    model.load_state_dict(best_state)

    if verbose:
        status = "early stopping" if stopped_early else "epochs agotados"
        print(
            f"[train_model] Finalizado ({status}). "
            f"best_val_loss={best_val_loss:.6f} en epoch {best_epoch}."
        )

    return {
        "model": model,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "history": history,
        "stopped_early": stopped_early,
    }