# src/anomaly/lstm_autoencoder.py
"""
Encoder LSTM, Decoder LSTM y modelo HybridLSTMAE completo para
detección de anomalías no supervisada sobre secuencias de embeddings bancarios.

Estructura del módulo
---------------------
    LSTMEncoder    — comprime la secuencia al cuello de botella (h_T, c_T)
    LSTMDecoder    — reconstruye la secuencia desde el cuello de botella
    HybridLSTMAE  — módulo raíz que orquesta MLP + Encoder + Decoder

Fundamentos teóricos: detección de anomalías por reconstrucción
---------------------------------------------------------------
Un autoencoder aprende a reconstruir su entrada. Si se entrena exclusivamente
sobre observaciones normales, aprende los patrones propios de la normalidad.
Ante una anomalía, el cuello de botella comprime y descarta la información
inusual, produciendo una reconstrucción cercana a la normalidad media.
El error de reconstrucción (MSE) actúa como anomaly score.

Tensión fundamental de diseño: si el cuello de botella tiene demasiada
capacidad, el modelo memoriza todo (MSE bajo tanto en normales como en
anomalías). Si tiene demasiado poca, no reconstruye ni los normales
(MSE alto para todo). El ratio de compresión 8:1 adoptado aquí es un
punto de partida dentro del rango recomendado por la literatura (6:1 a 12:1).

Encoder LSTM
------------
Procesa la secuencia proyectada E_proj = [e_proj_1, ..., e_proj_T] ∈ ℝ^(T × d_model)
paso a paso. El cuello de botella es el estado final (h_T, c_T):

    Encoder: ℝ^(T × d_model) → ℝ^(2 × lstm_hidden)

    c_T captura la memoria a largo plazo de la secuencia completa.
    h_T es la salida visible del encoder y actúa como representación comprimida.

Decoder LSTM (estrategia: entrada constante + reconstrucción invertida)
------------------------------------------------------------------------
De las tres variantes de decoder estudiadas para el TFM, se selecciona la
combinación "entrada constante + reconstrucción en orden inverso":

Variante descartada 1 — autoregresiva: la salida x_hat_t se usa como
    entrada en t+1. Acumula error en secuencias cortas (T=4) y hace el
    backpropagation más difícil.

Variante descartada 2 — secuencia invertida pura sin entrada constante:
    la información inicial varía en cada paso. Menos estable.

Variante seleccionada:
    - Entrada constante: h_T se replica T veces como entrada en cada paso.
      El decoder puede centrarse en "desplegar" la representación comprimida
      sin depender de reconstrucciones parciales previas.
    - Reconstrucción en orden inverso: el decoder produce
      [x_hat_T, x_hat_{T-1}, ..., x_hat_1].
      Justificación: el gradiente ∂L/∂h_T fluye directamente desde la primera
      posición del decoder (que corresponde a x_hat_T, el trimestre más
      reciente). Esto reduce la longitud del camino crítico de backpropagation
      de 2T-1 a T pasos, aliviando el problema de gradiente evanescente.

Cada estado oculto del decoder se proyecta al espacio d_model mediante
una capa lineal sin activación:

    x_hat_t = W_out · h_t^{dec} + b_out

Ratio de compresión
-------------------
Con T=4, d_model=96, lstm_hidden=48:
    ratio = (T × d_model) / lstm_hidden = (4 × 96) / 48 = 8:1

El cuello de botella (h_T, c_T) ∈ ℝ^(2 × 48) = 96 parámetros debe representar
una secuencia de 4 × 96 = 384 valores. Esta presión fuerza al encoder a
aprender únicamente los patrones estructurales de la normalidad.
"""

import torch
import torch.nn as nn
from src.models.mlp_projection import MLPProjection


# ---------------------------------------------------------------------------
# LSTMEncoder
# ---------------------------------------------------------------------------

class LSTMEncoder(nn.Module):
    """
    Encoder LSTM que comprime una secuencia proyectada al cuello de botella.

    Parámetros
    ----------
    d_model : int
        Dimensión de entrada por paso temporal (salida del MLPProjection).
    lstm_hidden : int
        Tamaño del estado oculto h_t y del estado de celda c_t.
    """

    def __init__(self, d_model: int = 96, lstm_hidden: int = 48) -> None:
        super().__init__()

        self.d_model = d_model
        self.lstm_hidden = lstm_hidden

        # batch_first=True: tensores con forma (batch, seq, features)
        # en lugar del formato legacy (seq, batch, features) de PyTorch.
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Procesa la secuencia y devuelve el estado final como cuello de botella.

        Parámetros
        ----------
        x : torch.Tensor
            Secuencia proyectada de forma (batch_size, T, d_model).

        Retorna
        -------
        h_T : torch.Tensor
            Estado oculto final, forma (batch_size, lstm_hidden).
            Actúa como representación comprimida de la secuencia.
        (h_T_raw, c_T) : tuple[torch.Tensor, torch.Tensor]
            Tupla de estados finales con forma (1, batch_size, lstm_hidden)
            cada uno. Se pasa al decoder para inicializar su estado.
        """
        # outputs: (batch, T, lstm_hidden) — ignorado, solo usamos el estado final
        # (h_n, c_n): (num_layers, batch, lstm_hidden) = (1, batch, lstm_hidden)
        _, (h_n, c_n) = self.lstm(x)

        # h_n tiene forma (1, batch, lstm_hidden); squeeze la dimensión de layers
        h_T = h_n.squeeze(0)  # (batch, lstm_hidden)

        return h_T, (h_n, c_n)


# ---------------------------------------------------------------------------
# LSTMDecoder
# ---------------------------------------------------------------------------

class LSTMDecoder(nn.Module):
    """
    Decoder LSTM con estrategia de entrada constante y reconstrucción invertida.

    El decoder recibe h_T replicado T veces como secuencia de entrada.
    Produce la reconstrucción en orden inverso: [x_hat_T, ..., x_hat_1].
    La capa de proyección lineal mapea cada estado oculto al espacio d_model.

    Parámetros
    ----------
    d_model : int
        Dimensión de la reconstrucción (debe coincidir con el encoder).
    lstm_hidden : int
        Tamaño del estado oculto (debe coincidir con el encoder).
    seq_len : int
        Longitud de la secuencia a reconstruir (T=4 por defecto).
    """

    def __init__(
        self,
        d_model: int = 96,
        lstm_hidden: int = 48,
        seq_len: int = 4,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.lstm_hidden = lstm_hidden
        self.seq_len = seq_len

        self.lstm = nn.LSTM(
            input_size=lstm_hidden,   # entrada = h_T replicado
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )

        # Proyección de cada estado oculto al espacio de reconstrucción.
        # Sin activación: la reconstrucción debe tener rango libre para
        # aproximar e_proj, que también tiene rango libre (salida del MLP).
        self.output_proj = nn.Linear(lstm_hidden, d_model)

    def forward(
        self,
        h_T: torch.Tensor,
        encoder_state: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        Reconstruye la secuencia desde el cuello de botella.

        Parámetros
        ----------
        h_T : torch.Tensor
            Estado oculto final del encoder, forma (batch_size, lstm_hidden).
        encoder_state : tuple[torch.Tensor, torch.Tensor]
            (h_n, c_n) del encoder con forma (1, batch_size, lstm_hidden).
            Se usa para inicializar el estado del decoder, transfiriendo
            tanto la memoria a corto plazo (h_n) como la memoria a largo
            plazo (c_n) del encoder al decoder.

        Retorna
        -------
        x_hat : torch.Tensor
            Reconstrucción en orden INVERTIDO, forma (batch_size, T, d_model).
            x_hat[:, 0, :] corresponde a x_hat_T (trimestre más reciente).
            x_hat[:, T-1, :] corresponde a x_hat_1 (trimestre más antiguo).
            IMPORTANTE: el caller (HybridLSTMAE y TemporalWeightedLoss) debe
            tener en cuenta este orden al aplicar pesos temporales.
        """
        batch_size = h_T.size(0)

        # Replicar h_T como entrada constante para cada paso del decoder.
        # h_T: (batch, lstm_hidden) → (batch, T, lstm_hidden)
        decoder_input = h_T.unsqueeze(1).expand(-1, self.seq_len, -1)

        # Inicializar decoder con el estado final del encoder.
        # El estado del decoder arranca donde terminó el encoder, lo que
        # proporciona una inicialización informativa en lugar de ceros.
        decoder_output, _ = self.lstm(decoder_input, encoder_state)
        # decoder_output: (batch, T, lstm_hidden)

        # Proyectar cada estado oculto al espacio d_model.
        x_hat = self.output_proj(decoder_output)
        # x_hat: (batch, T, d_model) — orden invertido

        return x_hat


# ---------------------------------------------------------------------------
# HybridLSTMAE
# ---------------------------------------------------------------------------

class HybridLSTMAE(nn.Module):
    """
    Autoencoder LSTM híbrido completo para detección de anomalías bancarias.

    Orquesta los tres componentes en secuencia:
        1. MLPProjection: proyecta [e_tab || e_rel] → e_proj en cada paso t
        2. LSTMEncoder: comprime E_proj = [e_proj_1..T] al cuello de botella
        3. LSTMDecoder: reconstruye E_proj desde el cuello de botella

    Flujo de datos (forward pass)
    -----------------------------
    Entrada: x ∈ ℝ^(batch, T, d_in)        [secuencia de vectores fusionados]
        ↓ MLPProjection (aplicado a cada t independientemente)
    e_proj ∈ ℝ^(batch, T, d_model)
        ↓ LSTMEncoder
    (h_T, c_T) ∈ ℝ^(2 × lstm_hidden)       [cuello de botella]
        ↓ LSTMDecoder
    x_hat ∈ ℝ^(batch, T, d_model)           [reconstrucción en orden inverso]

    El forward devuelve (e_proj, x_hat) para que la función de loss pueda
    calcular el MSE elemento a elemento sin depender de estado interno.

    Flujo de gradiente
    ------------------
    ∂L/∂x_hat → ∂L/∂decoder → ∂L/∂(h_T, c_T) → ∂L/∂encoder → ∂L/∂e_proj → ∂L/∂MLP

    Todo el modelo se optimiza conjuntamente con Adam. El MLP aprende la
    proyección que minimiza el error de reconstrucción del autoencoder
    completo, no una tarea auxiliar.

    Parámetros
    ----------
    d_in : int
        Dimensión de la entrada concatenada [e_tab || e_rel]. 256 por defecto.
    mlp_hidden : int
        Dimensión oculta del MLP de proyección. 192 por defecto.
    d_model : int
        Dimensión del espacio latente homogéneo (salida MLP, entrada LSTM). 96.
    lstm_hidden : int
        Dimensión del estado oculto del encoder y decoder LSTM. 48 por defecto.
    dropout : float
        Tasa de dropout en el MLP. 0.3 por defecto.
    seq_len : int
        Longitud de la ventana temporal T. 4 por defecto.
    """

    def __init__(
        self,
        d_in: int = 256,
        mlp_hidden: int = 192,
        d_model: int = 96,
        lstm_hidden: int = 48,
        dropout: float = 0.3,
        seq_len: int = 4,
    ) -> None:
        super().__init__()

        self.d_in = d_in
        self.d_model = d_model
        self.lstm_hidden = lstm_hidden
        self.seq_len = seq_len

        # Ratio de compresión: documentar explícitamente para verificación
        self.compression_ratio = (seq_len * d_model) / lstm_hidden

        self.mlp = MLPProjection(
            d_in=d_in,
            mlp_hidden=mlp_hidden,
            d_model=d_model,
            dropout=dropout,
        )

        self.encoder = LSTMEncoder(
            d_model=d_model,
            lstm_hidden=lstm_hidden,
        )

        self.decoder = LSTMDecoder(
            d_model=d_model,
            lstm_hidden=lstm_hidden,
            seq_len=seq_len,
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass completo del autoencoder híbrido.

        Parámetros
        ----------
        x : torch.Tensor
            Secuencia de vectores fusionados, forma (batch_size, T, d_in).
            Cada posición t contiene [e_tab_t || e_rel_t] del mismo banco.

        Retorna
        -------
        e_proj : torch.Tensor
            Representación proyectada, forma (batch_size, T, d_model).
            Es el "target" de la reconstrucción: la loss compara e_proj con x_hat.
        x_hat : torch.Tensor
            Reconstrucción del decoder en orden INVERTIDO,
            forma (batch_size, T, d_model).
            x_hat[:, 0, :] → reconstrucción del paso T (más reciente).
            x_hat[:, T-1, :] → reconstrucción del paso 1 (más antiguo).
        """
        # Paso 1: proyectar cada paso temporal con el MLP
        # El MLP opera sobre la última dimensión, PyTorch hace broadcasting
        # sobre (batch, T) automáticamente
        e_proj = self.mlp(x)  # (batch, T, d_model)

        # Paso 2: encoder LSTM — comprime la secuencia al cuello de botella
        h_T, encoder_state = self.encoder(e_proj)  # h_T: (batch, lstm_hidden)

        # Paso 3: decoder LSTM — reconstruye desde el cuello de botella
        x_hat = self.decoder(h_T, encoder_state)  # (batch, T, d_model), orden inverso

        return e_proj, x_hat

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Devuelve únicamente la representación comprimida h_T.
        Útil para inferencia y cálculo de anomaly score en producción.

        Parámetros
        ----------
        x : torch.Tensor
            Forma (batch_size, T, d_in).

        Retorna
        -------
        h_T : torch.Tensor
            Forma (batch_size, lstm_hidden).
        """
        e_proj = self.mlp(x)
        h_T, _ = self.encoder(e_proj)
        return h_T

    def get_compression_ratio(self) -> float:
        """Devuelve el ratio de compresión (T × d_model) / lstm_hidden."""
        return self.compression_ratio

    def count_parameters(self) -> dict[str, int]:
        """
        Cuenta los parámetros entrenables por componente.
        Útil para comparar configuraciones en la ablación.
        """
        def _count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        return {
            "mlp": _count(self.mlp),
            "encoder": _count(self.encoder),
            "decoder": _count(self.decoder),
            "total": _count(self),
        }

    def __repr__(self) -> str:
        ratio = self.get_compression_ratio()
        params = self.count_parameters()
        return (
            f"HybridLSTMAE(\n"
            f"  d_in={self.d_in}, d_model={self.d_model}, "
            f"lstm_hidden={self.lstm_hidden}, T={self.seq_len}\n"
            f"  compression_ratio={ratio:.1f}:1\n"
            f"  params_mlp={params['mlp']:,}, "
            f"params_encoder={params['encoder']:,}, "
            f"params_decoder={params['decoder']:,}\n"
            f"  total_params={params['total']:,}\n"
            f")"
        )