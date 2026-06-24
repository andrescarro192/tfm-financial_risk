# src/training/trainer_tgcn.py
"""
Funciones auxiliares para entrenamiento y evaluación del T-GCN.

Funciones:
    snapshot_to_device      : mueve los tensores de un snapshot al device indicado.
    compute_adj_dense       : convierte edge_index COO a matriz de adyacencia densa (N, N).
    _build_hidden_from_dict : construye hidden_state alineado por CERT, inicializando a cero los nodos nuevos.
    _update_hidden_dict     : vuelca el estado oculto actualizado al diccionario por CERT.
    evaluate                : evalúa el modelo sobre una secuencia de snapshots (AUROC, AUPRC, F1, precision, recall).
    find_best_threshold     : encuentra el umbral que maximiza F1 sobre validación.
    build_windows           : genera ventanas deslizantes de longitud W sobre la secuencia de snapshots.
    compute_val_loss        : calcula la loss media sobre validación, usada como criterio de early stopping.

Métricas relevantes con desbalanceo extremo (~0.044% positivos): AUROC, AUPRC y F1
con umbral optimizado. Accuracy descartada por trivialidad.
"""

import numpy as np
import torch
from torch_geometric.utils import to_dense_adj
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score
)
from src.models.tgcn import calculate_laplacian
import copy, time
import torch.nn as nn


def snapshot_to_device(snapshot, device):
    """
    Mueve los tensores de un snapshot al device indicado.
    Modifica el objeto in-place. Asumir que todos los snapshots
    van siempre al mismo device para evitar inconsistencias.
    """
    snapshot.x          = snapshot.x.to(device)
    snapshot.edge_index = snapshot.edge_index.to(device)
    snapshot.y          = snapshot.y.to(device)
    return snapshot


def compute_adj_dense(snapshot):
    """
    Convierte edge_index COO a matriz de adyacencia densa A^t ∈ R^{N×N}.
    to_dense_adj devuelve (1, N, N); hacemos squeeze para obtener (N, N).
    """
    adj = to_dense_adj(
        snapshot.edge_index,
        max_num_nodes=snapshot.num_nodes
    ).squeeze(0)  # (N, N)
    return adj


def _build_hidden_from_dict(cert_list, cert_to_hidden, hidden_dim, device):
    """
    Construye hidden_state (N, hidden_dim) alineado con cert_list.
    Nodos conocidos recuperan su vector del diccionario; nuevos se inicializan a cero.
    """
    h = torch.zeros(len(cert_list), hidden_dim, device=device)
    for i, cert in enumerate(cert_list):
        if cert in cert_to_hidden:
            h[i] = cert_to_hidden[cert]
    return h


def _update_hidden_dict(cert_list, hidden_state, cert_to_hidden):
    """Vuelca el estado oculto actualizado al diccionario por CERT."""
    for i, cert in enumerate(cert_list):
        cert_to_hidden[cert] = hidden_state[i].detach()


def evaluate(model, snapshots, device, threshold=0.5):
    """
    Evalúa el modelo sobre una secuencia de snapshots.

    Procesa la secuencia completa en modo inferencia y recoge
    probabilidades y etiquetas de TODOS los snapshots (no solo el último).
    Usa alineación por CERT para preservar el estado oculto de nodos
    persistentes entre trimestres.

    Returns dict con auroc, auprc, f1, precision, recall.
    """
    model.eval()
    all_probs  = []
    all_labels = []

    with torch.no_grad():
        cert_to_hidden = {}
        for snap in snapshots:
            snap    = snapshot_to_device(snap, device)
            adj     = compute_adj_dense(snap)   # (N, N)
            x       = snap.x                    # (N, d_x)
            y       = snap.y                    # (N,)
            cert_list = list(snap.cert)

            hidden_state = _build_hidden_from_dict(
                cert_list, cert_to_hidden, model.hidden_dim, device
            )

            laplacian    = calculate_laplacian(adj)
            hidden_state, _ = model.tgcn_cell(x, hidden_state, laplacian)
            _update_hidden_dict(cert_list, hidden_state, cert_to_hidden)

            logits       = model.classifier(hidden_state).squeeze(1)
            probs        = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    all_probs  = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    auroc  = roc_auc_score(all_labels, all_probs)
    auprc  = average_precision_score(all_labels, all_probs)
    preds  = (all_probs >= threshold).astype(int)
    f1     = f1_score(all_labels, preds, zero_division=0)
    prec   = precision_score(all_labels, preds, zero_division=0)
    rec    = recall_score(all_labels, preds, zero_division=0)

    return {'auroc': auroc, 'auprc': auprc, 'f1': f1,
            'precision': prec, 'recall': rec}


def find_best_threshold(model, snapshots, device, thresholds=None):
    """
    Encuentra el umbral de clasificación que maximiza F1 sobre
    el conjunto dado. Se usa SOLO sobre validación, nunca sobre test.
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        cert_to_hidden = {}
        for snap in snapshots:
            snap = snapshot_to_device(snap, device)
            adj  = compute_adj_dense(snap)
            cert_list = list(snap.cert)
            hidden_state = _build_hidden_from_dict(
                cert_list, cert_to_hidden, model.hidden_dim, device
            )
            laplacian       = calculate_laplacian(adj)
            hidden_state, _ = model.tgcn_cell(snap.x, hidden_state, laplacian)
            _update_hidden_dict(cert_list, hidden_state, cert_to_hidden)
            logits          = model.classifier(hidden_state).squeeze(1)
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(snap.y.cpu().numpy())

    all_probs  = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    best_f1, best_thr = 0.0, 0.5
    for thr in thresholds:
        f1 = f1_score(all_labels, (all_probs >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    return best_thr, best_f1

def build_windows(snapshots: list, W: int) -> list[list]:
    """
    Genera ventanas deslizantes de longitud W sobre la secuencia de snapshots.

    Con 17 snapshots y W=8 produce 10 ventanas:
      [0:8], [1:9], [2:10], ..., [9:17]

    Cada ventana es una sub-secuencia contigua que el T-GCN procesa
    independientemente con estado oculto reinicializado a cero.
    """
    return [snapshots[i:i+W] for i in range(len(snapshots) - W + 1)]


def compute_val_loss(model, snapshots, criterion, device):
    """
    Calcula la loss media sobre todos los snapshots de validación.
    Se usa como criterio de early stopping en lugar de AUROC,
    más estable con pocos positivos.
    """
    model.eval()
    total_loss   = 0.0

    with torch.no_grad():
        cert_to_hidden = {}
        for snap in snapshots:
            snap = snapshot_to_device(snap, device)
            adj  = compute_adj_dense(snap)
            cert_list = list(snap.cert)
            hidden_state = _build_hidden_from_dict(
                cert_list, cert_to_hidden, model.hidden_dim, device
            )

            laplacian       = calculate_laplacian(adj)
            hidden_state, _ = model.tgcn_cell(snap.x, hidden_state, laplacian)
            _update_hidden_dict(cert_list, hidden_state, cert_to_hidden)
            logits          = model.classifier(hidden_state).squeeze(1)
            loss            = criterion(logits, snap.y.float())
            total_loss     += loss.item()

    return total_loss / len(snapshots)

# Funcion principal de entrenamiento

def train_tgcn(model, optimizer, scheduler, criterion,
               snapshots_train, snapshots_val,
               W, MAX_EPOCHS, PATIENCE, DEVICE):
    """
    Bucle de entrenamiento del T-GCN con early stopping sobre val_loss.
    Usa ventanas deslizantes de longitud W y alineación de estado oculto
    por CERT dentro de cada ventana (cert_to_hidden_window).
    Devuelve el modelo con los pesos del mejor epoch y el historial de métricas.
    """

    windows = build_windows(snapshots_train, W)
    print(f'Ventanas deslizantes: {len(windows)} (W={W}, sobre {len(snapshots_train)} snapshots train)')

    history = {
        'train_loss': [], 'val_loss': [], 'val_auroc': [], 'val_auprc': [], 'lr': []
    }

    best_val_loss  = float('inf')
    best_epoch     = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    no_improv      = 0
    t0             = time.perf_counter()

    for epoch in range(1, MAX_EPOCHS + 1):

        model.train()
        epoch_loss = 0.0

        for window in windows:
            optimizer.zero_grad()
            cert_to_hidden_window = {}
            window_loss = 0.0

            for snap in window:
                snap      = snapshot_to_device(snap, DEVICE)
                adj       = compute_adj_dense(snap)
                cert_list = list(snap.cert)

                hidden_state = _build_hidden_from_dict(
                    cert_list, cert_to_hidden_window, model.hidden_dim, DEVICE
                )
                hidden_state = hidden_state.detach()

                laplacian       = calculate_laplacian(adj)
                hidden_state, _ = model.tgcn_cell(snap.x, hidden_state, laplacian)
                _update_hidden_dict(cert_list, hidden_state, cert_to_hidden_window)

                logits      = model.classifier(hidden_state).squeeze(1)
                loss        = criterion(logits, snap.y.float())
                window_loss += loss

            window_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += window_loss.item()

        avg_train_loss = epoch_loss / len(windows)

        val_loss    = compute_val_loss(model, snapshots_val, criterion, DEVICE)
        val_metrics = evaluate(model, snapshots_val, DEVICE)

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['val_auroc'].append(val_metrics['auroc'])
        history['val_auprc'].append(val_metrics['auprc'])
        history['lr'].append(current_lr)

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_epoch     = epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            no_improv      = 0
        else:
            no_improv += 1

        if epoch % 25 == 0 or epoch == 1:
            elapsed = time.perf_counter() - t0
            print(
                f'Epoch {epoch:4d}/{MAX_EPOCHS} | '
                f'train_loss={avg_train_loss:.4f} | '
                f'val_loss={val_loss:.4f} | '
                f'val_auroc={val_metrics["auroc"]:.4f} | '
                f'val_auprc={val_metrics["auprc"]:.4f} | '
                f'lr={current_lr:.6f} | '
                f'elapsed={elapsed:.0f}s'
            )

        if no_improv >= PATIENCE:
            print(f'\nEarly stopping en epoch {epoch}. '
                  f'Mejor epoch: {best_epoch} (val_loss={best_val_loss:.4f})')
            break

    model.load_state_dict(best_model_wts)
    print(f'\nEntrenamiento completado. Mejor epoch: {best_epoch} | val_loss: {best_val_loss:.4f}')

    return model, history