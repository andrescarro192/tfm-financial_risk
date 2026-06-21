# src/models/lstm_baseline.py
"""
LSTM Baseline supervisado para predicción de quiebra bancaria.

Responsabilidad única: recibir una secuencia temporal de embeddings
tabulares e_tab ∈ ℝ^(T, 192) por entidad y producir un logit de
probabilidad de quiebra en el último trimestre de la ventana (period_end).

Rol en el diseño experimental
------------------------------
Este modelo es el término de comparación "solo datos numéricos" de la
hipótesis central del TFM: ¿añadir información relacional (e_rel, vía
T-GCN) mejora la capacidad predictiva de quiebra respecto a un modelo que
solo ve información tabular (e_tab, vía TabPFN)?

Para que esa comparación sea válida, este Baseline y el futuro modelo
Híbrido deben ser arquitectónica y metodológicamente idénticos en todo
excepto en la dimensión de entrada (192 frente a 256). Cualquier asimetría
adicional —profundidad distinta, régimen de entrenamiento distinto,
optimización desigual de hiperparámetros— introduce una variable de
confusión que invalida la atribución de cualquier diferencia de rendimiento
a la presencia de información relacional. Por esta razón la arquitectura
se fija aquí de antemano y NO se ajusta de forma posterior persiguiendo el
mejor resultado posible: un baseline que se optimiza ad-hoc deja de
cumplir su función de referencia.

Arquitectura
------------
    LSTM(input_size=d_in, hidden_size=lstm_hidden, num_layers=1, batch_first=True)
    Dropout(dropout)
    Linear(lstm_hidden, 1)   # logit crudo, sin sigmoid

Se usa el hidden state del último paso temporal (equivalente al último
trimestre de la ventana, period_end) como resumen de la secuencia:
h_T = lstm_out[:, -1, :]. Esto es coherente con la definición de
is_anomalous/failed ya usada en la construcción de secuencias (Bloque 2):
la etiqueta se asocia al estado de la entidad en el trimestre de cierre
de la ventana, no a pasos intermedios.

Por qué una sola capa LSTM y dropout simple
----------------------------------------------
Con un universo de aproximadamente 70 positivos en el bloque de desarrollo
(2016Q1-2021Q4), el riesgo dominante no es la falta de capacidad expresiva
del modelo sino el sobreajuste a esas pocas instancias positivas
específicas. Una arquitectura más profunda (múltiples capas LSTM, cabezas
de clasificación con capas ocultas intermedias) aumenta la complejidad de
la clase de hipótesis sin que haya evidencia de que la tarea lo requiera,
y reduce la cantidad de datos disponibles por parámetro a ajustar de forma
relevante para la clase minoritaria.

El argumento formal es una versión de la cota de generalización vía
complejidad de Rademacher (Bartlett & Mendelson, 2002): el error de
generalización crece con la complejidad de la clase de hipótesis G y
decrece con el tamaño de muestra efectivo. En clasificación con
desbalanceo extremo, el tamaño de muestra "efectivo" para la clase
positiva es el que limita la capacidad de generalizar sobre esa clase,
no el total de observaciones (que está dominado por negativos). Esto
justifica mantener la arquitectura deliberadamente conservadora: una
capa LSTM, dropout como único mecanismo de regularización adicional, y
una proyección lineal directa a logit sin capas intermedias.

El espacio de búsqueda de hiperparámetros (lr, dropout, weight_decay,
pos_weight factor) se explora vía grid search/Optuna sobre estos valores
de entrenamiento, no sobre la topología, siguiendo el mismo procedimiento
ya validado en el T-GCN.

Por qué BCEWithLogitsLoss con pos_weight, no MSE ni Focal Loss (de momento)
------------------------------------------------------------------------------
Con una tasa de positivos de desarrollo en torno al 0.05-0.1% (70 positivos
sobre miles de observaciones-trimestre), BCE sin ponderar converge
trivialmente a predecir siempre la clase negativa: el gradiente agregado
de los positivos es despreciable frente al de los negativos. pos_weight
amplifica la contribución al gradiente de cada positivo en proporción
w = pw_factor * (n_neg / n_pos). replicando el procedimiento ya validado
en el T-GCN (pw_factor=0.25 demostrado mejor que 0.5 y que el ratio bruto
sin atenuar). El modelo se queda con el logit crudo (sin sigmoid) porque
BCEWithLogitsLoss aplica internamente la formulación numéricamente estable
de log-sigmoid, evitando overflow/underflow para logits extremos.

Focal Loss queda como experimento posterior de ajuste fino, no como punto
de partida del Baseline, para no introducir un grado de libertad adicional
(el hiperparámetro gamma de focal loss) en la primera comparación.

Inicialización
--------------
PyTorch inicializa nn.LSTM con una distribución uniforme U(-1/sqrt(H), 1/sqrt(H))
por defecto, adecuada para la mayoría de los casos y se mantiene sin
modificar. La capa de salida (Linear) usa Xavier/Glorot uniforme con
ganancia unitaria, apropiada para una capa sin no linealidad posterior
(el logit es la salida cruda del modelo).
"""

import numpy as np
import torch
import torch.nn as nn


class LSTMBaseline(nn.Module):
    """
    LSTM supervisado para clasificación binaria de riesgo de insolvencia
    sobre trayectorias temporales de embeddings compactos.

    Parámetros
    ----------
    d_in : int
        Dimensión de entrada por paso temporal. 192 para el Baseline (solo e_tab).
    lstm_hidden : int
        Dimensión del espacio de estados ocultos del LSTM.
    dropout : float
        Tasa de abandono (dropout) aplicada sobre el vector terminal h_last.
    num_layers : int
        Número de capas LSTM apiladas. Por defecto 1.
    pos_prevalence : float, opcional
        Tasa de prevalencia empírica de la clase anómala en el conjunto de
        entrenamiento (34 / 66922 ≈ 0.000508). Se utiliza para inicializar
        el sesgo del clasificador y calibrar el prior inicial.
    """

    def __init__(
        self,
        d_in: int = 192,
        lstm_hidden: int = 32,
        dropout: float = 0.3,
        num_layers: int = 1,
        pos_prevalence: float = 0.000508,
    ) -> None:
        super().__init__()

        self.d_in = d_in
        self.lstm_hidden = lstm_hidden
        self.dropout_rate = dropout
        self.num_layers = num_layers
        self.pos_prevalence = pos_prevalence

        self.lstm = nn.LSTM(
            input_size=d_in,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0,  # Desactivado inter-capa al ser num_layers=1
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Inicialización avanzada de pesos y sesgos para mitigar los efectos
        del desequilibrio extremo de clases sobre el gradiente inicial.
        """
        # 1. Inicialización Canónica de Xavier para la matriz del clasificador
        nn.init.xavier_uniform_(self.classifier.weight, gain=1.0)

        # 2. Focal Bias Tuning: Inicialización del sesgo de salida según el Prior Real
        # b = ln(p / (1 - p)) evita pérdidas iniciales desproporcionadas por falsos positivos
        if self.pos_prevalence > 0:
            initial_bias = np.log(self.pos_prevalence / (1.0 - self.pos_prevalence))
            nn.init.constant_(self.classifier.bias, initial_bias)
        else:
            nn.init.zeros_(self.classifier.bias)

        # 3. Inicialización Avanzada de las Compuertas del nn.LSTM
        # Forzar el sesgo de la compuerta de olvido (forget gate) a 1.0 (Jozefowicz et al., 2015)
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias_tensor = getattr(self.lstm, name)
                n = bias_tensor.size(0)
                # El bias de PyTorch en LSTM concatena [b_i, b_f, b_g, b_o], cada uno de tamaño lstm_hidden
                # Colocamos a 1.0 la sección correspondiente a la compuerta de olvido (segundo bloque)
                with torch.no_grad():
                    bias_tensor.fill_(0.0)  # Reset a cero de base
                    bias_tensor[self.lstm_hidden : 2 * self.lstm_hidden].fill_(
                        1.0
                    )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Mapea el tensor secuencial de entrada hacia el logit lineal del horizonte terminal.

        Parámetros
        ----------
        x : torch.Tensor
            Tensor de dimensiones (batch_size, T, d_in), donde T=4 trimestres.

        Retorna
        -------
        logits : torch.Tensor
            Logit crudo de quiebra de forma (batch_size,). Sin activar vía sigmoid.
        """
        lstm_out, _ = self.lstm(x)  # lstm_out: (B, T, lstm_hidden)

        # Extracción equivariante del estado oculto final (period_end)
        h_last = lstm_out[:, -1, :]  # (B, lstm_hidden)

        # Regularización y proyección lineal hacia el espacio de decisión
        h_last = self.dropout(h_last)
        logits = self.classifier(h_last).squeeze(-1)  # (B,)

        return logits

    def __repr__(self) -> str:
        return (
            f"LSTMBaseline("
            f"d_in={self.d_in}, "
            f"lstm_hidden={self.lstm_hidden}, "
            f"num_layers={self.num_layers}, "
            f"dropout={self.dropout_rate}, "
            f"pos_prevalence={self.pos_prevalence})"
        )