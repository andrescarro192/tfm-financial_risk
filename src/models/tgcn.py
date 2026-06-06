# src/models/tgcn.py
'''

 T-GCN adaptado para clasificación binaria nodo a nodo sobre grafos dinámicos.

 Diferencias respecto a la implementación de referencia (Zhao et al. 2019):
   1. Input multidimensional: x_i^t ∈ R^{input_dim} en vez de escalar.
      La concatenación [x, h] pasa de (N, d_h+1) a (N, d_h+d_x), y W ∈
      R^{(d_h+d_x) × d_h} en consecuencia.
   2. Laplaciano dinámico: L̂^t se calcula en cada paso temporal del forward
      porque A^t varía por snapshot. No se registra en __init__.
   3. Clasificación binaria: cabeza Linear(hidden_dim, 1) + sigmoid en vez
      de regresor. Salida ŷ_i^t ∈ [0,1] por nodo.
   4. Sin pytorch-lightning ni batch dimension: cada snapshot es un grafo
      completo, no hay batch_size > 1.

 Notación:
   N   — número de nodos en el snapshot actual (varía por trimestre)
   d_x — input_dim = 204 (192 embedding TabPFN + 12 atributos estructurales)
   d_h — hidden_dim (hiperparámetro, punto de partida: 64)
   T   — longitud de la secuencia de snapshots procesada en un forward pass
'''

import torch
import torch.nn as nn


'''
 Utilidad: normalización simétrica del Laplaciano
 -----------------------------------------------------------------------------
 Dado A^t ∈ {0,1}^{N×N}, calcula:
   Ã = A^t + I                          (auto-conexiones)
   D̃_ii = Σ_j Ã_ij                     (grados de Ã)
   L̂ = D̃^{-1/2} Ã D̃^{-1/2}           (normalización simétrica Kipf & Welling)

 Para nodos aislados (sin vecinos en A^t), Ã_ii = 1 y D̃_ii = 1,
 por lo que L̂_ii = 1: la agregación GCN degenera en la identidad sobre x_i^t,
 es decir, el nodo solo se transforma a sí mismo sin propagar señal de vecinos.
 Esto es semánticamente correcto: un banco sin holding no recibe contagio
 estructural, solo proyecta sus propias features.
'''

def calculate_laplacian(adj: torch.Tensor) -> torch.Tensor:
    """
    Calcula L̂ = D̃^{-1/2} (A + I) D̃^{-1/2}.

    Args:
        adj: matriz de adyacencia A^t ∈ R^{N×N}, dtype float, en el device correcto.

    Returns:
        L̂ ∈ R^{N×N}, mismo device que adj.
    """
    n = adj.size(0)
    a_tilde = adj + torch.eye(n, device=adj.device, dtype=adj.dtype)
    row_sum = a_tilde.sum(dim=1)                          # D̃ diagonal
    d_inv_sqrt = torch.pow(row_sum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0             # nodos aislados: 0 seguro
    # L̂ = D̃^{-1/2} Ã D̃^{-1/2}  equivale a escalar filas y columnas
    laplacian = d_inv_sqrt.unsqueeze(1) * a_tilde * d_inv_sqrt.unsqueeze(0)
    return laplacian


'''
 TGCNGraphConvolution — capa GCN dentro de la celda GRU
 -----------------------------------------------------------------------------
 Implementa una única operación de la forma:
   out = L̂ [x, h] W + b
 donde [x, h] es la concatenación horizontal de input y estado oculto.

 En la celda T-GCN esta operación aparece dos veces:
   - graph_conv1: calcula [r, u] con output_dim = 2 * hidden_dim
   - graph_conv2: calcula el candidato c̃ con output_dim = hidden_dim

 Shape tracking (sin batch dimension, un snapshot a la vez):
   inputs        : (N, d_x)
   hidden_state  : (N, d_h)
   concat        : (N, d_x + d_h)    ← diferencia clave respecto al repo original
   L̂ @ concat   : (N, d_x + d_h)
   output        : (N, output_dim)   después de W ∈ R^{(d_x+d_h) × output_dim}'''
class TGCNGraphConvolution(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, bias: float = 0.0):
        """
        Args:
            input_dim:  d_x, dimensión de los atributos de nodo x_i^t.
            hidden_dim: d_h, dimensión del estado oculto h_i^t.
            output_dim: dimensión de salida (2*d_h para puertas r,u; d_h para c̃).
            bias:       valor inicial del bias (1.0 para puertas, 0.0 para candidato).
        """
        super().__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # W ∈ R^{(d_h + d_x) × output_dim}
        # La concatenación es [h, x] de dimensión (d_h + d_x) por nodo.
        self.weights = nn.Parameter(
            torch.FloatTensor(hidden_dim + input_dim, output_dim)
        )
        self.biases = nn.Parameter(torch.FloatTensor(output_dim))
        self._bias_init = bias
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.biases, self._bias_init)

    def forward(self, inputs: torch.Tensor, hidden_state: torch.Tensor,
                laplacian: torch.Tensor) -> torch.Tensor:
        """
        Computa L̂ [x, h] W + b.

        Args:
            inputs      : (N, d_x)
            hidden_state: (N, d_h)
            laplacian   : (N, N)  L̂^t ya calculado para este snapshot

        Returns:
            (N, output_dim)
        """
        # [h, x] → (N, d_h + d_x)
        concat = torch.cat([hidden_state, inputs], dim=1)
        # L̂ [h, x] → (N, d_h + d_x)  — propagación sobre el grafo
        aggregated = laplacian @ concat
        # (N, d_h + d_x) @ (d_h + d_x, output_dim) + b → (N, output_dim)
        return aggregated @ self.weights + self.biases


'''
 TGCNCell — celda GRU con convolución sobre grafo
 -----------------------------------------------------------------------------
 Extiende la GRU estándar sustituyendo las transformaciones lineales por
 convoluciones GCN. Las ecuaciones de actualización son:

   z_t = σ( GCN([h_{t-1}, x_t], A^t) )    puerta de actualización
   r_t = σ( GCN([h_{t-1}, x_t], A^t) )    puerta de reset
   c̃_t = tanh( GCN([r_t ⊙ h_{t-1}, x_t], A^t) )   candidato
   h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ c̃_t

 graph_conv1 computa [z, r] conjuntamente (output_dim = 2*d_h) y luego
 se separan con torch.chunk, igual que en la GRU estándar.
 graph_conv2 computa c̃ (output_dim = d_h).

 El Laplaciano L̂^t se pasa como argumento desde el forward de TGCN,
 donde se calcula una sola vez por snapshot antes del loop temporal.'''
class TGCNCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Args:
            input_dim:  d_x = 204
            hidden_dim: d_h, hiperparámetro.
        """
        super().__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        # Puertas z y r conjuntas: output_dim = 2 * d_h
        # bias=1.0 inicializa las puertas cerca de 1 → al inicio el modelo
        # tiende a conservar el estado anterior, comportamiento estable.
        self.graph_conv1 = TGCNGraphConvolution(
            input_dim, hidden_dim, output_dim=hidden_dim * 2, bias=1.0
        )
        # Candidato c̃: output_dim = d_h
        self.graph_conv2 = TGCNGraphConvolution(
            input_dim, hidden_dim, output_dim=hidden_dim, bias=0.0
        )

    def forward(self, inputs: torch.Tensor, hidden_state: torch.Tensor,
                laplacian: torch.Tensor):
        """
        Un paso temporal de la celda T-GCN.

        Args:
            inputs      : (N, d_x)   atributos nodales en t
            hidden_state: (N, d_h)   estado oculto en t-1
            laplacian   : (N, N)     L̂^t para este snapshot

        Returns:
            output      : (N, d_h)   = new_hidden_state
            new_hidden_state: (N, d_h)
        """
        # [z, r] = σ( GCN([h, x], A^t) )  →  (N, 2*d_h)
        zr = torch.sigmoid(self.graph_conv1(inputs, hidden_state, laplacian))
        # z: puerta actualización, r: puerta reset  →  cada (N, d_h)
        z, r = torch.chunk(zr, chunks=2, dim=1)

        # c̃ = tanh( GCN([r⊙h, x], A^t) )  →  (N, d_h)
        c = torch.tanh(self.graph_conv2(inputs, r * hidden_state, laplacian))

        # h_t = (1-z)⊙h_{t-1} + z⊙c̃
        new_hidden_state = (1.0 - z) * hidden_state + z * c

        return new_hidden_state, new_hidden_state


'''
 TGCN — modelo completo
 -----------------------------------------------------------------------------
 Procesa una secuencia de snapshots {G^1, ..., G^T} y devuelve el embedding
 dinámico h_i^T ∈ R^{d_h} de cada nodo al final de la secuencia.

 A diferencia del repo original, aquí:
   - No hay batch dimension: cada snapshot es un grafo completo.
   - El número de nodos N varía entre snapshots (quiebras, fusiones).
     El estado oculto h se reinicializa a cero al principio de cada secuencia.
   - El Laplaciano L̂^t se calcula dinámicamente para cada snapshot.
   - La cabeza clasificadora proyecta h_i^T → ŷ_i ∈ [0,1].
'''

class TGCN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Args:
            input_dim:  d_x = 204, dimensión de x_i^t.
            hidden_dim: d_h, dimensión del estado oculto GRU.
        """
        super().__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        self.tgcn_cell  = TGCNCell(input_dim, hidden_dim)

        # Cabeza clasificadora: h_i^T ∈ R^{d_h} → ŷ_i ∈ [0,1]
        # Linear(d_h, 1) + sigmoid. En inferencia se usa sigmoid;
        # durante el entrenamiento con BCEWithLogitsLoss se puede omitir
        # el sigmoid para estabilidad numérica (ver notebook 06).
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, snapshots: list) -> tuple:
        """
        Procesa una secuencia de snapshots y devuelve logits y etiquetas
        del último snapshot (el que se supervisa en cada paso de entrenamiento).

        Args:
            snapshots: lista de torch_geometric.data.Data, ordenados temporalmente.
                       Cada elemento tiene:
                         .x         (N_t, d_x)   atributos de nodo
                         .adj       (N_t, N_t)   matriz de adyacencia densa
                         .y         (N_t,)        etiquetas binarias
                         .num_nodes int

        Returns:
            logits: (N_T,)   logits del último snapshot (sin sigmoid)
            y:      (N_T,)   etiquetas del último snapshot
        """
        hidden_state = None

        for snapshot in snapshots:
            x   = snapshot.x                          # (N_t, d_x)
            adj = snapshot.adj                        # (N_t, N_t)
            n   = snapshot.num_nodes

            # Inicializar o reinicializar hidden_state si N cambia entre snapshots.
            # Cuando N_t ≠ N_{t-1} (quiebra o entrada de nuevos bancos), el estado
            # oculto de nodos que ya no existen se descarta y los nuevos arrancan en 0.
            # En la práctica se reconstruye el estado completo para N_t desde cero
            # porque el mapping de índices entre trimestres está en meta_t, no aquí.
            # El notebook 06 gestiona la alineación de nodos entre snapshots.
            if hidden_state is None or hidden_state.size(0) != n:
                hidden_state = torch.zeros(n, self.hidden_dim, device=x.device)

            # Laplaciano dinámico para este snapshot: O(N^2) pero N~5000, viable.
            laplacian = calculate_laplacian(adj)      # (N_t, N_t)

            # Un paso T-GCN
            hidden_state, _ = self.tgcn_cell(x, hidden_state, laplacian)
            # hidden_state: (N_t, d_h)

        # Cabeza clasificadora sobre el último snapshot
        logits = self.classifier(hidden_state).squeeze(1)   # (N_T,)
        y      = snapshots[-1].y                             # (N_T,)

        return logits, y

    def predict_proba(self, snapshots: list) -> torch.Tensor:
        """
        Devuelve probabilidades ŷ_i = σ(logit_i) ∈ [0,1].
        Usar solo en inferencia, no durante el entrenamiento.
        """
        logits, _ = self.forward(snapshots)
        return torch.sigmoid(logits)

    @property
    def hyperparameters(self) -> dict:
        return {"input_dim": self.input_dim, "hidden_dim": self.hidden_dim}