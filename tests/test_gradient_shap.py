"""
test_gradient_shap.py

Prueba mínima, aislada y desechable. NO usa datos ni checkpoints reales.
Objetivo único: verificar que shap.GradientExplainer acepta un input 3D
(batch, T=4, d_in=192) sobre un LSTMBaseline y devuelve algo con forma
y aditividad sensatas, antes de tocar el pipeline real.

Tres comprobaciones, en orden de prioridad:
  1. ¿GradientExplainer acepta el input 3D sin romper?
  2. ¿La forma del output es (N, 4, 192), sin lista envolvente?
  3. ¿Los valores son sensatos? (no NaN, no todo-cero, aditividad aproximada)
"""

import numpy as np
import torch
import torch.nn as nn
import shap
import traceback


# ---------------------------------------------------------------------------
# 0. Recrear LSTMBaseline de forma mínima (misma arquitectura, pesos random)
# ---------------------------------------------------------------------------

class LSTMBaseline(nn.Module):
    def __init__(self, d_in=192, lstm_hidden=32, dropout=0.3, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=d_in,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        h_last = lstm_out[:, -1, :]
        h_last = self.dropout(h_last)
        logits = self.classifier(h_last).squeeze(-1)
        return logits


class LSTMBaselineSHAPWrapper(nn.Module):
    """
    Envuelve LSTMBaseline únicamente para satisfacer la convención de shape
    2D (batch, 1) que shap.GradientExplainer espera del output del modelo
    (internamente indexa outputs[:, idx]). No reentrena ni modifica pesos:
    delega el forward íntegro al modelo original y solo añade una dimensión
    final mediante unsqueeze. La causa raíz del IndexError observado es que
    LSTMBaseline.forward hace .squeeze(-1) y devuelve (batch,), 1D puro.
    """
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        logits = self.base_model(x)   # (batch,)
        return logits.unsqueeze(-1)   # (batch, 1)


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    device = "cpu"
    d_in, T = 192, 4

    model = LSTMBaseline(d_in=d_in, lstm_hidden=32, dropout=0.3, num_layers=1)
    model.eval()  # crítico: dropout debe estar desactivado para SHAP

    wrapped_model = LSTMBaselineSHAPWrapper(model)
    wrapped_model.eval()

    # --- Datos sintéticos ---
    n_background = 20
    n_inputs = 5

    background = torch.randn(n_background, T, d_in, dtype=torch.float32)
    inputs = torch.randn(n_inputs, T, d_in, dtype=torch.float32)

    print("=" * 70)
    print("PRUEBA 1: ¿GradientExplainer acepta input 3D sin romper?")
    print("=" * 70)
    try:
        explainer = shap.GradientExplainer(wrapped_model, background)
        print("OK: GradientExplainer construido sin error.")
    except Exception:
        print("FALLO al construir GradientExplainer:")
        traceback.print_exc()
        return

    try:
        shap_values = explainer.shap_values(inputs, nsamples=50)
        print("OK: shap_values() no lanzó excepción.\n")
    except Exception:
        print("FALLO en shap_values():")
        traceback.print_exc()
        return

    print("=" * 70)
    print("PRUEBA 2: ¿La forma del output es (N, 4, 192) sin lista envolvente?")
    print("=" * 70)
    is_list = isinstance(shap_values, list)
    print(f"Tipo devuelto: {type(shap_values)}")
    if is_list:
        print(f"  -> Es una lista de longitud {len(shap_values)}.")
        print(f"  -> shap_values[0] tiene forma: {np.array(shap_values[0]).shape}")
        sv = np.array(shap_values[0])
    else:
        sv = np.array(shap_values)
        print(f"  -> Array directo, forma: {sv.shape}")
    
    #introducimos un guardarrail
    if sv.shape[-1] == 1:
        sv = sv.squeeze(-1)

    expected_shape = (n_inputs, T, d_in)
    shape_ok = sv.shape == expected_shape
    print(f"Forma esperada: {expected_shape} | Forma obtenida: {sv.shape} | "
          f"{'OK' if shape_ok else 'MISMATCH'}\n")

    print("=" * 70)
    print("PRUEBA 3: ¿Valores sensatos? (NaN, ceros, aditividad aproximada)")
    print("=" * 70)
    has_nan = np.isnan(sv).any()
    all_zero = np.allclose(sv, 0.0)
    print(f"¿Contiene NaN?: {has_nan}")
    print(f"¿Todo cero?: {all_zero}")

    # Aditividad aproximada: phi.sum() + E[f(background)] ≈ f(input)
    with torch.no_grad():
        f_inputs = model(inputs).numpy()  # (n_inputs,)
        f_background_mean = model(background).numpy().mean()  # escalar

    phi_sum = sv.sum(axis=(1, 2))  # (n_inputs,) suma de todos los SHAP values por muestra
    reconstructed = phi_sum + f_background_mean
    diff = np.abs(reconstructed - f_inputs)

    print("\nChequeo de aditividad (por muestra):")
    print(f"{'logit real':>12} {'reconstruido':>14} {'diff abs':>10}")
    for i in range(n_inputs):
        print(f"{f_inputs[i]:12.4f} {reconstructed[i]:14.4f} {diff[i]:10.4f}")

    print(f"\nDiferencia media absoluta: {diff.mean():.4f}")
    print(f"Diferencia máxima absoluta: {diff.max():.4f}")
    print("(GradientSHAP es una aproximación estocástica: se espera diferencia")
    print(" pequeña pero no nula. Si la diferencia es grande o errática, subir")
    print(" nsamples y repetir antes de confiar en los resultados.)")

    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70)
    print(f"Input 3D aceptado:        {'SI' if True else 'NO'}")
    print(f"Forma correcta:           {'SI' if shape_ok else 'NO'}")
    print(f"Sin NaN:                  {'SI' if not has_nan else 'NO'}")
    print(f"No todo-cero:             {'SI' if not all_zero else 'NO'}")
    print(f"Devuelve lista envolvente: {'SI' if is_list else 'NO'} "
          f"({'el código de compute_gradient_shap necesita la salvaguarda' if is_list else 'la salvaguarda del código no se activa, pero no estorba'})")


if __name__ == "__main__":
    main()