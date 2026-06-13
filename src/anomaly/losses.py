# src/anomaly/losses.py
"""
Función de pérdida para el LSTM Autoencoder híbrido con soporte de
ponderación temporal fija sobre los pasos de reconstrucción.

Motivación teórica
------------------
En early warning bancario, el trimestre más reciente de una ventana temporal
es el más informativo para la detección del riesgo de quiebra. Una loss MSE
uniforme trata todos los pasos temporales como igualmente importantes, lo que
puede llevar al encoder a comprimir información equilibradamente cuando lo
deseable es priorizar la reconstrucción del estado más reciente.

La loss ponderada introduce pesos fijos crecientes:

    w = [0.1, 0.2, 0.3, 0.4]    para t = [1, 2, 3, 4]

La suma de pesos es 1.0, por lo que la escala de la loss es comparable entre
la variante uniforme (w_t = 0.25 para todo t) y la ponderada. Esto es crítico
para que la comparación en la ablación sea justa.

Implicación sobre el gradiente
-------------------------------
Con ponderación temporal, el gradiente de h_T recibe una contribución cuatro
veces mayor del trimestre t=4 (más reciente) que del trimestre t=1 (más
antiguo). Esto especializa h_T en capturar los patrones necesarios para
reconstruir el estado reciente de la secuencia.

    ∂L/∂h_T = Σ_t w_t * ∂(MSE_t)/∂h_T

Para t=4: contribución de peso 0.4. Para t=1: contribución de peso 0.1.

Interacción crítica con reconstrucción invertida
------------------------------------------------
El decoder produce x_hat en orden INVERTIDO:
    decoder_step=0 → x_hat_T  (trimestre más reciente de la secuencia original)
    decoder_step=1 → x_hat_{T-1}
    ...
    decoder_step=T-1 → x_hat_1

Por tanto, los pesos deben aplicarse también en orden invertido sobre el
tensor x_hat:
    w_decoder = [0.4, 0.3, 0.2, 0.1]

Aplicar w=[0.1, 0.2, 0.3, 0.4] directamente sobre el tensor del decoder
sería un error silencioso: penalizaría más la reconstrucción del trimestre
más ANTIGUO (que aparece en la última posición del decoder).

Este módulo gestiona este detalle internamente: el caller solo necesita
pasar los pesos en orden "lógico" w=[0.1, 0.2, 0.3, 0.4] y la clase
aplica la inversión correcta.

MSE vs MAE
----------
MSE (seleccionado): penaliza cuadráticamente los errores grandes, amplificando
la señal de anomalías con errores de reconstrucción elevados. Justificado
teóricamente para detección de anomalías donde los outliers tienen errores
notablemente mayores.

MAE (alternativa en ablación): penaliza linealmente, más robusto a outliers
en entrenamiento pero menos sensible a anomalías extremas en evaluación.
Se incluye como opción pero no es la variante principal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TemporalWeightedLoss(nn.Module):
    """
    Loss MSE con ponderación temporal opcional sobre pasos de reconstrucción.

    Soporta dos modos de operación controlados por use_temporal_weighting:

    Modo uniforme (use_temporal_weighting=False):
        L = (1/T) * Σ_t MSE(e_proj_t, x_hat_t)
        Equivale a MSE estándar promediado sobre los T pasos.

    Modo ponderado (use_temporal_weighting=True):
        L = Σ_t w_t * MSE(e_proj_t, x_hat_t)
        donde w_t son pesos fijos con Σ_t w_t = 1.

    En ambos casos el MSE de cada paso se promedia sobre d_model antes de
    aplicar los pesos. Esto desacopla la escala de la loss de la dimensión
    del embedding, haciendo los valores comparables entre configuraciones
    de ablación con distintos d_model.

    Parámetros
    ----------
    weights : list[float] | None
        Pesos temporales en orden LÓGICO [w_1, w_2, ..., w_T].
        Ejemplo: [0.1, 0.2, 0.3, 0.4] para T=4.
        Si None, se usa distribución uniforme.
        El módulo invierte internamente los pesos para coherencia con
        la reconstrucción en orden inverso del decoder.
    use_temporal_weighting : bool
        Si True, aplica los pesos temporales. Si False, usa pesos uniformes
        independientemente del valor de 'weights'.
    reduction : str
        'mean': promedia sobre el batch (comportamiento por defecto).
        'sum': suma sobre el batch.
        'none': devuelve loss por elemento del batch.
    """

    def __init__(
        self,
        weights: Optional[list[float]] = None,
        use_temporal_weighting: bool = True,
        reduction: str = "mean",
    ) -> None:
        super().__init__()

        self.use_temporal_weighting = use_temporal_weighting
        self.reduction = reduction

        if weights is not None:
            w = torch.tensor(weights, dtype=torch.float32)

            # Verificar que los pesos suman 1.0 (tolerancia numérica)
            w_sum = w.sum().item()
            if abs(w_sum - 1.0) > 1e-5:
                raise ValueError(
                    f"Los pesos temporales deben sumar 1.0, pero suman {w_sum:.6f}. "
                    f"Normaliza los pesos antes de pasarlos: weights = [w/sum(w) for w in weights]"
                )

            # CRÍTICO: invertir los pesos para coherencia con reconstrucción invertida.
            # El decoder produce x_hat en orden [x_hat_T, x_hat_{T-1}, ..., x_hat_1].
            # Si weights=[0.1, 0.2, 0.3, 0.4] en orden lógico (t=1 a T),
            # entonces sobre el tensor del decoder deben aplicarse [0.4, 0.3, 0.2, 0.1].
            # Guardar ambas versiones para diagnóstico.
            self.register_buffer("weights_logical", w)                    # [0.1, 0.2, 0.3, 0.4]
            self.register_buffer("weights_decoder", w.flip(dims=[0]))     # [0.4, 0.3, 0.2, 0.1]
        else:
            self.register_buffer("weights_logical", None)
            self.register_buffer("weights_decoder", None)

    def forward(
        self,
        e_proj: torch.Tensor,
        x_hat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calcula la loss entre la representación proyectada y la reconstrucción.

        Parámetros
        ----------
        e_proj : torch.Tensor
            Representación proyectada (target), forma (batch_size, T, d_model).
            Proviene directamente del MLPProjection.
        x_hat : torch.Tensor
            Reconstrucción del decoder en orden INVERTIDO,
            forma (batch_size, T, d_model).
            x_hat[:, 0, :] corresponde a x_hat_T (más reciente).
            x_hat[:, T-1, :] corresponde a x_hat_1 (más antiguo).
            IMPORTANTE: e_proj debe estar también en orden original (t=1..T).
            Esta función invierte e_proj internamente para comparar
            correctamente con x_hat.

        Retorna
        -------
        loss : torch.Tensor
            Escalar (si reduction='mean' o 'sum') o tensor (batch,)
            (si reduction='none').
        """
        T = x_hat.size(1)

        # PASO 1: alinear e_proj con el orden del decoder.
        # x_hat está en orden inverso [t=T, T-1, ..., 1].
        # e_proj está en orden original [t=1, 2, ..., T].
        # Invertimos e_proj para que la comparación sea elemento a elemento correcta.
        e_proj_inv = e_proj.flip(dims=[1])  # ahora también en orden [t=T, ..., t=1]

        # PASO 2: MSE por paso temporal, promediado sobre d_model.
        # error: (batch, T, d_model)
        error = (e_proj_inv - x_hat) ** 2

        # mse_per_step: (batch, T) — error promedio sobre la dimensión de embedding
        mse_per_step = error.mean(dim=-1)

        # PASO 3: aplicar pesos temporales.
        if self.use_temporal_weighting and self.weights_decoder is not None:
            # weights_decoder está en orden [w_T, w_{T-1}, ..., w_1]
            # que coincide con el orden del decoder, por lo que la
            # multiplicación es directa.
            w = self.weights_decoder  # (T,) en orden del decoder
            if w.size(0) != T:
                raise ValueError(
                    f"Número de pesos ({w.size(0)}) distinto de la longitud "
                    f"de la secuencia T={T}. Verifica WINDOW_LENGTH y weights."
                )
            # Broadcast: (batch, T) * (T,) → (batch, T)
            weighted_mse = mse_per_step * w
            # Suma sobre T (los pesos ya suman 1, así que esta suma es la loss)
            loss_per_sample = weighted_mse.sum(dim=-1)  # (batch,)
        else:
            # Loss uniforme: promedio simple sobre T
            loss_per_sample = mse_per_step.mean(dim=-1)  # (batch,)

        # PASO 4: reducción sobre el batch
        if self.reduction == "mean":
            return loss_per_sample.mean()
        elif self.reduction == "sum":
            return loss_per_sample.sum()
        elif self.reduction == "none":
            return loss_per_sample
        else:
            raise ValueError(f"reduction debe ser 'mean', 'sum' o 'none', no '{self.reduction}'")

    def compute_step_scores(
        self,
        e_proj: torch.Tensor,
        x_hat: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Calcula anomaly scores desagregados por paso temporal.

        Este método se usa exclusivamente en evaluación (Bloque 5), nunca
        durante entrenamiento. Los scores paso a paso se calculan siempre
        con pesos UNIFORMES (mse_t puro), independientemente de
        use_temporal_weighting. La razón es que la ponderación distorsionaría
        la interpretación de la anticipación temporal: un trimestre con MSE
        moderado pero peso alto produciría un score ponderado alto que no
        refleja un error de reconstrucción grave.

        Parámetros
        ----------
        e_proj : torch.Tensor
            (batch, T, d_model) en orden original (t=1..T).
        x_hat : torch.Tensor
            (batch, T, d_model) en orden invertido (t=T..1).

        Retorna
        -------
        dict con:
            'score_global': (batch,) — MSE promedio ponderado (coherente con train)
            'score_t': (batch, T) — MSE puro por paso en orden original [t=1..T]
            'w_score_t': (batch, T) — MSE ponderado por paso en orden original
                         (solo si use_temporal_weighting=True, si no es None)
        """
        # Invertir e_proj para comparar con x_hat (orden del decoder)
        e_proj_inv = e_proj.flip(dims=[1])

        error = (e_proj_inv - x_hat) ** 2
        mse_per_step_decoder = error.mean(dim=-1)  # (batch, T) en orden decoder

        # Reordenar al orden lógico original (t=1..T) para la salida
        mse_per_step_original = mse_per_step_decoder.flip(dims=[1])  # (batch, T)

        # Score global (coherente con el entrenamiento)
        if self.use_temporal_weighting and self.weights_logical is not None:
            w = self.weights_logical  # (T,) en orden lógico
            score_global = (mse_per_step_original * w).sum(dim=-1)  # (batch,)
            w_score_t = mse_per_step_original * w                    # (batch, T)
        else:
            score_global = mse_per_step_original.mean(dim=-1)        # (batch,)
            w_score_t = None

        return {
            "score_global": score_global,          # (batch,) — para AUC-PR, umbral
            "score_t": mse_per_step_original,      # (batch, T) — para anticipación temporal
            "w_score_t": w_score_t,                # (batch, T) o None
        }

    def __repr__(self) -> str:
        if self.use_temporal_weighting and self.weights_logical is not None:
            w_log = self.weights_logical.tolist()
            w_dec = self.weights_decoder.tolist()
            return (
                f"TemporalWeightedLoss(\n"
                f"  use_temporal_weighting=True\n"
                f"  weights_logical (t=1..T): {w_log}\n"
                f"  weights_decoder (decoder order): {w_dec}\n"
                f"  reduction='{self.reduction}'\n"
                f")"
            )
        else:
            return (
                f"TemporalWeightedLoss("
                f"use_temporal_weighting=False, "
                f"reduction='{self.reduction}')"
            )


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_loss(
    use_temporal_weighting: bool,
    weights: Optional[list[float]] = None,
    reduction: str = "mean",
) -> TemporalWeightedLoss:
    """
    Construye la función de loss según los parámetros del Bloque 0.

    Se usa tanto en el entrenamiento base (Bloque 4) como en cada
    ejecución de la ablación (Bloque 7), evitando duplicación de lógica
    de construcción.

    Parámetros
    ----------
    use_temporal_weighting : bool
        USE_TEMPORAL_WEIGHTING del Bloque 0.
    weights : list[float] | None
        W_T del Bloque 0 = [0.1, 0.2, 0.3, 0.4].
        Si None y use_temporal_weighting=True, se usa distribución uniforme.
    reduction : str
        'mean' para entrenamiento, 'none' para evaluación batch a batch.

    Retorna
    -------
    TemporalWeightedLoss
    """
    if use_temporal_weighting and weights is None:
        # Fallback: uniforme si se activa ponderación sin especificar pesos
        T = 4
        weights = [1.0 / T] * T

    return TemporalWeightedLoss(
        weights=weights,
        use_temporal_weighting=use_temporal_weighting,
        reduction=reduction,
    )