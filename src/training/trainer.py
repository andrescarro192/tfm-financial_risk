# src/trainer.py
"""
Funciones de entrenamiento, evaluación y early stopping, agnósticas al
modelo concreto (reciben model como argumento). Reutilizables sin cambios
para el Baseline y para el futuro Híbrido, ya que ambos comparten la
misma arquitectura y el mismo protocolo de entrenamiento.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, precision_recall_curve
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train_one_epoch(model, loader, optimizer, criterion, clip_norm, device):
    model.train()
    total_loss = 0.0
    n_obs = 0
    for batch in loader:
        x = batch["X"].to(device)
        y = batch["failed"].to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        n_obs += x.size(0)

    return total_loss / n_obs


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """
    Evalúa sobre el split completo, no por batch. Necesario porque con
    pocos positivos repartidos en muchos batches, una métrica de ranking
    calculada batch a batch no tiene sentido estadístico: hay que
    concatenar logits y targets de todo el split antes de calcular
    AUC-PR/AUC-ROC.
    """
    model.eval()
    total_loss = 0.0
    n_obs = 0
    all_logits, all_targets = [], []

    for batch in loader:
        x = batch["X"].to(device)
        y = batch["failed"].to(device)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        n_obs += x.size(0)
        all_logits.append(logits.cpu())
        all_targets.append(y.cpu())

    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    probs = torch.sigmoid(all_logits).numpy()
    targets = all_targets.numpy()

    auc_pr = average_precision_score(targets, probs)
    auc_roc = roc_auc_score(targets, probs)

    return total_loss / n_obs, auc_pr, auc_roc


def train_with_early_stopping(
    model,
    dataloader_train,
    dataloader_val,
    pos_weight_raw: float,
    pw_factor: float = 1.0,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    clip_norm: float = 1.0,
    max_epochs: int = 100,
    patience: int = 15,
    scheduler_factor: float = 0.5,
    scheduler_patience: int = 5,
    device: str = "cpu",
    verbose: bool = True,
):
    """
    Restauración de mejor época: se conserva una copia de los pesos cada
    vez que val_auc_pr mejora, y al final se restaura esa copia, no los
    pesos de la última época ejecutada.

    Scheduler: ReduceLROnPlateau monitoreando val_loss (no val_auc_pr).
    val_loss es una señal continua sobre todo el split de validación;
    val_auc_pr, con solo unos pocos positivos en val, es una señal de
    baja resolución y poco adecuada para gobernar la reducción de
    learning rate. scheduler_patience se fija por debajo de patience
    (early stopping) para permitir varias reducciones de lr antes de
    agotar la paciencia y detener el entrenamiento.
    """
    pos_weight_used = torch.tensor(
        pw_factor * pos_weight_raw, dtype=torch.float32, device=device
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_used)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=scheduler_factor, patience=scheduler_patience
    )

    best_auc_pr = -float("inf")
    best_state = None
    epochs_sin_mejora = 0
    history = []

    model.to(device)

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(
            model, dataloader_train, optimizer, criterion, clip_norm, device
        )
        val_loss, val_auc_pr, val_auc_roc = evaluate(
            model, dataloader_val, criterion, device
        )

        lr_antes = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        lr_actual = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auc_pr": val_auc_pr,
            "val_auc_roc": val_auc_roc,
            "lr": lr_actual,
        })

        if verbose:
            msg = (
                f"Epoch {epoch:3d} | train_loss={train_loss:.4f} "
                f"| val_loss={val_loss:.4f} | val_AUC-PR={val_auc_pr:.4f} "
                f"| val_AUC-ROC={val_auc_roc:.4f} | lr={lr_actual:.2e}"
            )
            if lr_actual < lr_antes:
                msg += f"  <- lr reducido ({lr_antes:.2e} -> {lr_actual:.2e})"
            print(msg)

        if val_auc_pr > best_auc_pr:
            best_auc_pr = val_auc_pr
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_sin_mejora = 0
        else:
            epochs_sin_mejora += 1

        if epochs_sin_mejora >= patience:
            if verbose:
                print(
                    f"\nEarly stopping en epoch {epoch}: sin mejora en "
                    f"val_AUC-PR durante {patience} epochs."
                )
            break

    model.load_state_dict(best_state)
    return model, history, best_auc_pr



def plot_training_history(history: list, title: str = "Entrenamiento LSTM Baseline") -> None:
    """
    Grafica train_loss/val_loss y val_AUC-PR/val_AUC-ROC por epoch.
    Marca con línea vertical el epoch de mejor val_AUC-PR, que es el que
    train_with_early_stopping usa para restaurar los pesos finales
    (best_state), no necesariamente el último epoch ejecutado.
    """
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_auc_pr = [h["val_auc_pr"] for h in history]
    val_auc_roc = [h["val_auc_roc"] for h in history]

    best_idx = max(range(len(history)), key=lambda i: val_auc_pr[i])
    best_epoch = epochs[best_idx]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].plot(epochs, train_loss, label="train_loss")
    axes[0].plot(epochs, val_loss, label="val_loss")
    axes[0].axvline(best_epoch, color="gray", linestyle="--", alpha=0.6,
                     label=f"mejor epoch ({best_epoch})")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss de entrenamiento y validación")
    axes[0].legend()

    axes[1].plot(epochs, val_auc_pr, label="val_AUC-PR", color="tab:green")
    axes[1].plot(epochs, val_auc_roc, label="val_AUC-ROC", color="tab:orange")
    axes[1].axvline(best_epoch, color="gray", linestyle="--", alpha=0.6,
                     label=f"mejor epoch ({best_epoch})")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Métrica")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Métricas de validación")
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    plt.show()



import json
from pathlib import Path


def train_multi_seed_and_save(
    model_cls,
    model_kwargs: dict,
    best_params: dict,
    dataloader_train,
    dataloader_val,
    pos_weight_raw: float,
    seeds: list,
    output_dir: str,
    device: str = "cpu",
    max_epochs: int = 100,
    patience: int = 15,
    scheduler_factor: float = 0.5,
    scheduler_patience: int = 5,
    clip_norm: float = 1.0,
):
    """
    Reentrena la configuración ganadora sobre cada semilla en `seeds`,
    persiste los pesos de la mejor época de cada una y su histórico.
    Si las semillas coinciden con las usadas dentro del objective de
    Optuna para ese mismo trial, el best_auc_pr de cada semilla aquí
    debería reproducir casi exactamente el valor guardado en
    trial.user_attrs["auc_pr_per_seed"]; si no coincide, hay una fuente
    de aleatoriedad no controlada que conviene investigar antes de
    confiar en los pesos exportados.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    resultados = []

    for seed in seeds:
        torch.manual_seed(seed)

        model = model_cls(**model_kwargs)

        model, history, best_auc_pr = train_with_early_stopping(
            model=model,
            dataloader_train=dataloader_train,
            dataloader_val=dataloader_val,
            pos_weight_raw=pos_weight_raw,
            pw_factor=best_params["pw_factor"],
            lr=best_params["lr"],
            weight_decay=best_params["weight_decay"],
            clip_norm=clip_norm,
            max_epochs=max_epochs,
            patience=patience,
            scheduler_factor=scheduler_factor,
            scheduler_patience=scheduler_patience,
            device=device,
            verbose=True,
        )

        weights_path = output_path / f"lstm_baseline_seed{seed}.pt"
        history_path = output_path / f"history_seed{seed}.json"

        torch.save(model.state_dict(), weights_path)
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        resultados.append({
            "seed": seed,
            "best_auc_pr": best_auc_pr,
            "weights_path": str(weights_path),
            "history_path": str(history_path),
        })
        print(f"Semilla {seed}: val_AUC-PR={best_auc_pr:.4f} -> {weights_path.name}")

    metadata_path = output_path / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump({
            "best_params": best_params,
            "model_kwargs": model_kwargs,
            "pos_weight_raw": pos_weight_raw,
            "seeds": seeds,
            "resultados": resultados,
        }, f, indent=2)

    return resultados




def load_ensemble(model_cls, model_kwargs: dict, weights_paths: list, device: str = "cpu"):
    models = []
    for path in weights_paths:
        model = model_cls(**model_kwargs)
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        models.append(model)
    return models

