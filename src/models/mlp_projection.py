# src/models/mlp_projection.py
"""
Módulo de proyección multimodal para el LSTM Autoencoder híbrido.

Responsabilidad única: recibir el vector concatenado [e_tab || e_rel] ∈ ℝ^d_in
y proyectarlo a un espacio latente homogéneo e_proj ∈ ℝ^d_model.

Motivación teórica
------------------
e_tab (192 dims) proviene del espacio de atención de TabPFN; e_rel (64 dims)
proviene del espacio espectral del T-GCN. Aunque ambos representan el mismo
banco en el mismo trimestre, sus distribuciones internas son heterogéneas:
distintas escalas, distintos radios en el espacio latente, distintas semánticas
geométricas. Concatenar directamente y pasar al LSTM sobrecargaría la red con
dos tareas simultáneas: (1) alinear distribuciones heterogéneas y (2) modelar
dinámica temporal. La capa MLP separa ambas responsabilidades.

Arquitectura
------------
    Linear(d_in → mlp_hidden)   [256 → 192 por defecto]
    ReLU()
    Dropout(dropout)
    Linear(mlp_hidden → d_model) [192 → 96 por defecto]
    # Sin activación en la salida: el LSTM necesita rango no acotado

Inicialización
--------------
Ambas capas lineales usan inicialización Xavier/Glorot uniforme, que es
óptima para capas seguidas de ReLU bajo la hipótesis de varianza unitaria
de activaciones. Los sesgos se inicializan a cero.

Entrenamiento conjunto
----------------------
Este módulo NO se entrena de forma independiente. Sus parámetros se optimizan
conjuntamente con el encoder y decoder LSTM minimizando el error de
reconstrucción total. El gradiente fluye:
    decoder → encoder → MLP (este módulo)
Esto garantiza que la proyección aprendida sea óptima para la tarea de
reconstrucción temporal, no para una tarea auxiliar genérica.
"""

import torch
import torch.nn as nn


class MLPProjection(nn.Module):
    """
    MLP de dos capas que proyecta el vector fusionado [e_tab || e_rel]
    al espacio d_model del encoder LSTM.

    Parámetros
    ----------
    d_in : int
        Dimensión de entrada. d_tab + d_rel = 192 + 64 = 256 por defecto.
    mlp_hidden : int
        Dimensión de la capa oculta. 192 por defecto (mismo orden que d_tab).
    d_model : int
        Dimensión de salida. Entrada al encoder LSTM. 96 por defecto.
    dropout : float
        Tasa de dropout aplicada tras la activación ReLU. 0.3 por defecto.
    """

    def __init__(
        self,
        d_in: int = 256,
        mlp_hidden: int = 192,
        d_model: int = 96,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.d_in = d_in
        self.mlp_hidden = mlp_hidden
        self.d_model = d_model

        self.net = nn.Sequential(
            nn.Linear(d_in, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, d_model),
            # Sin activación final: rango libre para el LSTM
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Inicialización Xavier/Glorot uniforme para las capas lineales.

        Xavier uniforme escala los pesos según:
            W ~ U(-a, a)  donde  a = gain * sqrt(6 / (fan_in + fan_out))

        Para ReLU se usa gain = sqrt(2) (He initialization).
        Para la capa de salida sin activación se usa gain = 1.0.
        Los sesgos se inicializan a cero en ambas capas.
        """
        layers = [m for m in self.net if isinstance(m, nn.Linear)]

        # Capa oculta: seguida de ReLU → gain = sqrt(2)
        nn.init.xavier_uniform_(layers[0].weight, gain=nn.init.calculate_gain("relu"))
        nn.init.zeros_(layers[0].bias)

        # Capa de salida: sin activación → gain = 1.0
        nn.init.xavier_uniform_(layers[1].weight, gain=1.0)
        nn.init.zeros_(layers[1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Proyección del vector fusionado al espacio d_model.

        Parámetros
        ----------
        x : torch.Tensor
            Tensor de forma (batch_size, d_in) o (batch_size, T, d_in).
            Si se recibe una secuencia (B, T, d_in), el MLP opera sobre
            cada paso temporal de forma independiente. Esto es válido porque
            nn.Linear aplica la misma transformación lineal a la última
            dimensión (broadcasting implícito de PyTorch).

        Retorna
        -------
        torch.Tensor
            Tensor de forma (batch_size, d_model) o (batch_size, T, d_model),
            preservando las dimensiones de batch y tiempo si las hay.
        """
        return self.net(x)

    def __repr__(self) -> str:
        return (
            f"MLPProjection("
            f"d_in={self.d_in}, "
            f"mlp_hidden={self.mlp_hidden}, "
            f"d_model={self.d_model})"
        )