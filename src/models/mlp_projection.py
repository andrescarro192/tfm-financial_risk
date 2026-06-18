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
    LayerNorm(d_model)
    # Sin activación tras LayerNorm: el LSTM necesita rango no acotado

Adición: LayerNorm tras la proyección (corrección de colapso de representación)
---------------------------------------------------------------------------------
Diagnóstico empírico (notebook 06, análisis de separabilidad por etapas):
entrenado end-to-end con MLP+LSTM minimizando solo MSE de reconstrucción,
el MLP converge a una salida e_proj de varianza casi nula (varianza total
≈0.0067, frente a ≈0.78-1.42 en los embeddings de entrada e_rel/e_tab).
Esto es un caso de "representation collapse": el decoder LSTM, con
capacidad suficiente, puede reconstruir aproximadamente un e_proj casi
constante desde cualquier estado oculto, por lo que el sistema completo
encuentra un mínimo de MSE en el que e_proj colapsa a un volumen
extremadamente pequeño del espacio ℝ^d_model. La consecuencia es que
TODA la información de entrada -incluida la señal casi perfectamente
discriminativa presente en e_rel (AUC-ROC=1.00 sobre e_rel crudo)- se
pierde (AUC-ROC≈0.53, equivalente a azar, en e_proj y en h_T).

LayerNorm(d_model) normaliza cada vector e_proj_i individualmente a media 0
y varianza unitaria por componente (con parámetros afines aprendibles
gamma, beta). Esto hace que el colapso de varianza deje de reducir el MSE:
si el MLP produjera una salida pre-norm casi constante, LayerNorm
amplificaría su (pequeña) varianza relativa hasta escala unitaria,
produciendo una reconstrucción peor, no mejor. El colapso deja de ser un
mínimo favorable de la loss.

LayerNorm normaliza por vector (última dimensión), no por batch, por lo
que es compatible con el broadcasting sobre (batch, T, d_in) y no depende
del tamaño de batch ni introduce dependencia entre muestras (relevante para
inferencia con batch=1 y para la ablación con distintos BATCH_SIZE).

No contradice la justificación de "sin activación en la salida, rango
libre": LayerNorm renormaliza la ESCALA de la salida, no acota su RANGO
como lo haría una activación tipo sigmoid/tanh. El vector normalizado
sigue pudiendo tomar cualquier valor real; solo se fija su media y
varianza por vector, y los parámetros afines gamma/beta permiten al
modelo reescalar si fuera óptimo.

Inicialización
--------------
Ambas capas lineales usan inicialización Xavier/Glorot uniforme, que es
óptima para capas seguidas de ReLU bajo la hipótesis de varianza unitaria
de activaciones. Los sesgos se inicializan a cero. LayerNorm se inicializa
con sus valores por defecto de PyTorch (gamma=1, beta=0), que ya
corresponden a "no reescalar tras la normalización" -el punto de partida
neutro.

Entrenamiento conjunto
----------------------
Este módulo NO se entrena de forma independiente. Sus parámetros (incluidos
gamma y beta de LayerNorm) se optimizan conjuntamente con el encoder y
decoder LSTM minimizando el error de reconstrucción total. El gradiente
fluye:
    decoder → encoder → LayerNorm → Linear(2) → MLP (este módulo)
Esto garantiza que la proyección aprendida sea óptima para la tarea de
reconstrucción temporal, no para una tarea auxiliar genérica.
"""

import torch
import torch.nn as nn


class MLPProjection(nn.Module):
    """
    MLP de dos capas con LayerNorm de salida, que proyecta el vector
    fusionado [e_tab || e_rel] al espacio d_model del encoder LSTM.

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
            # Sin activación tras la proyección lineal: rango libre para el LSTM.
            # LayerNorm normaliza ESCALA (media/varianza por vector), no RANGO:
            # previene el colapso de varianza de e_proj sin acotar sus valores.
            nn.LayerNorm(d_model),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Inicialización de pesos para las capas lineales.

        Capa oculta:
            Se utiliza Kaiming/He Uniforme (diseñada específicamente para capas 
            seguidas de activaciones ReLU). Escala los pesos según una distribución:
            W ~ U(-bound, bound) donde bound = sqrt(6 / fan_in)
            Esto previene la atenuación del gradiente en las etapas iniciales.

        Capa de salida:
            Se utiliza Xavier/Glorot Uniforme (sin activación exógena).
            W ~ U(-a, a) donde a = 1.0 * sqrt(6 / (fan_in + fan_out))
            
        Los sesgos de las capas lineales se inicializan a cero de forma explícita.

        LayerNorm se deja con su inicialización por defecto de PyTorch
        (gamma=1, beta=0): punto de partida neutro, sin reescalado.
        """
        layers = [m for m in self.net if isinstance(m, nn.Linear)]

        # Capa oculta: seguida de ReLU → Inicialización Kaiming/He Uniforme
        # mode='fan_in' preserva la magnitud de las variaciones en el pase forward
        nn.init.kaiming_uniform_(layers[0].weight, nonlinearity='relu')
        nn.init.zeros_(layers[0].bias)

        # Capa de salida: sin activación exógena → Ganancia unitaria (Xavier)
        nn.init.xavier_uniform_(layers[1].weight, gain=1.0)
        nn.init.zeros_(layers[1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Proyección del vector fusionado al espacio d_model, normalizada.

        Parámetros
        ----------
        x : torch.Tensor
            Tensor de forma (batch_size, d_in) o (batch_size, T, d_in).
            Si se recibe una secuencia (B, T, d_in), el MLP y LayerNorm
            operan sobre cada paso temporal de forma independiente. Esto
            es válido porque tanto nn.Linear como nn.LayerNorm(d_model)
            aplican su transformación a la última dimensión, con
            broadcasting de PyTorch sobre las dimensiones anteriores.

        Retorna
        -------
        torch.Tensor
            Tensor de forma (batch_size, d_model) o (batch_size, T, d_model),
            con media ≈0 y varianza ≈1 por componente a lo largo de la
            última dimensión (antes del reescalado afín gamma/beta de
            LayerNorm), preservando las dimensiones de batch y tiempo si
            las hay.
        """
        return self.net(x)

    def __repr__(self) -> str:
        return (
            f"MLPProjection("
            f"d_in={self.d_in}, "
            f"mlp_hidden={self.mlp_hidden}, "
            f"d_model={self.d_model}, "
            f"layer_norm=True)"
        )