"""
src/evaluation/interpretability.py

Módulo de interpretabilidad para el ensamble LSTM Baseline mediante GradientSHAP.

Justificación del explainer:
    GradientSHAP calcula gradientes esperados sobre interpolaciones aleatorias
    entre el input y el background. A diferencia de DeepSHAP/DeepLIFT, que
    sustituyen el backward estándar por reglas de atribución personalizadas
    (problema documentado en arquitecturas recurrentes), GradientSHAP usa
    autograd ordinario, por lo que no hereda ese problema con nn.LSTM. Opera
    sobre logits crudos (el forward del modelo no aplica sigmoid), lo que
    garantiza gradientes no saturados y aditividad exacta en espacio log-odds.

    Nota de implementación — wrapper de salida:
        shap.GradientExplainer asume internamente una salida 2D del modelo,
        (batch, n_outputs), e indexa esa segunda dimensión. LSTMBaseline.forward
        devuelve (batch,) tras un squeeze(-1) explícito, lo que provoca un
        IndexError dentro de shap (_gradient.py, método gradient). Para no
        modificar la arquitectura entrenada, se usa LSTMBaselineSHAPWrapper,
        que delega el forward íntegro y solo añade unsqueeze(-1) para exponer
        (batch, 1). Como consecuencia, shap_values() devuelve un array con una
        dimensión final unitaria (N, 4, 192, 1) en vez de (N, 4, 192) directo;
        compute_gradient_shap() la elimina con squeeze(-1) tras verificar ndim.
        Verificado empíricamente en test_gradient_shap.py antes de aplicarse
        sobre datos reales.

Decisiones metodológicas fijadas:
    Background dataset:
        100 secuencias de bancos sanos (failed=0) del bloque de ENTRENAMIENTO.
        Motivo 1 — coherencia distributiva: el modelo aprendió la normalidad
        sobre el bloque de entrenamiento; usar esa misma referencia evita
        introducir distribuciones ajenas al proceso de aprendizaje.
        Motivo 2 — pureza analítica: el background representa estabilidad
        financiera pura. Cualquier desviación positiva en el análisis local
        mide la fuerza patológica de deterioro respecto a la normalidad del
        sistema, no respecto a un promedio que ya incluye entidades colapsadas.
        Motivo 3 — data leakage: el bloque de evaluación contiene los 33
        positivos que se analizan; incluirlo en el background contaminaría
        la referencia con información de las propias muestras a explicar.
        100 instancias es el estándar de la literatura para modelar la
        distribución base con coste computacional controlado.

    Muestra global:
        2.000 secuencias aleatorias del bloque de evaluación (negativas y
        positivas en su proporción natural). Una muestra n=2.000 sobre
        N=74.231 garantiza margen de error < 2.2% con confianza del 95%
        bajo máxima varianza. La caracterización arquitectónica es
        estadísticamente equivalente al censo completo.

    Muestra local:
        33 secuencias positivas íntegras (9 CERTs únicos, etiqueta propagada).
        No se submuestrea: es el conjunto completo de interés regulatorio.

Estructura del análisis:
    - Global:  2.000 secuencias, absolute=True (importancia neta |φ|).
               Responde qué posición temporal concentra más señal discriminativa
               en promedio sobre el panel.
    - Local:   33 secuencias positivas, absolute=False (signo preservado).
               Permite ver si cada trimestre empuja hacia quiebra (+) o
               retiene el modelo (-). Conecta con trayectorias y rankings.
"""

import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import shap
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from typing import Optional



# Wrapper para paliar el problema dimensional de SHAP
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
    
# ---------------------------------------------------------------------------
# 0. Función auxiliar: construcción del background y la muestra global
# ---------------------------------------------------------------------------

def build_shap_inputs(
    sequences_train: list,
    sequences_eval: list,
    n_background: int = 100,
    n_global_sample: int = 2000,
    seed: int = 42,
    device: str = "cpu",
) -> dict:
    """
    Construye los tensores de entrada para los dos análisis SHAP.

    Background:
        100 secuencias de bancos sanos (failed=0) del bloque de entrenamiento.
        Representa la normalidad financiera aprendida por el modelo. Usar el
        bloque de entrenamiento evita data leakage respecto al bloque de
        evaluación que contiene los positivos a analizar.

    Muestra global:
        2.000 secuencias aleatorias del bloque de evaluación en su proporción
        natural (negativos y positivos). Estadísticamente equivalente al
        censo completo con margen de error < 2.2% al 95% de confianza.

    Muestra local:
        33 secuencias con failed=1 del bloque de evaluación, íntegras.
        No se submuestrea.

    Args:
        sequences_train:  Lista de dicts del bloque de entrenamiento.
                          Cada dict contiene al menos 'X' (array o tensor
                          de forma (4, 192)) y 'failed' (int 0/1).
        sequences_eval:   Lista de dicts del bloque de evaluación.
                          Cada dict contiene 'X', 'failed', 'CERT',
                          'period_end'.
        n_background:     Número de secuencias sanas para el background.
                          Por defecto 100.
        n_global_sample:  Número de secuencias para el análisis global.
                          Por defecto 2.000.
        seed:             Semilla para reproducibilidad del muestreo.
        device:           Dispositivo PyTorch ('cpu' o 'cuda').

    Returns:
        Dict con claves:
            'background':       Tensor (100, 4, 192) en float32.
            'global_inputs':    Tensor (2000, 4, 192) en float32.
            'global_labels':    np.ndarray (2000,) con etiquetas failed.
            'global_certs':     Lista (2000,) de CERT.
            'global_periods':   Lista (2000,) de period_end.
            'local_inputs':     Tensor (33, 4, 192) en float32.
            'local_labels':     np.ndarray (33,) con etiquetas failed.
            'local_certs':      Lista (33,) de CERT.
            'local_periods':    Lista (33,) de period_end.
    """
    rng = np.random.default_rng(seed)

    # --- Background: sanos del bloque de entrenamiento ---
    healthy_train = [s for s in sequences_train if s["failed"] == 0]
    if len(healthy_train) < n_background:
        raise ValueError(
            f"El bloque de entrenamiento tiene solo {len(healthy_train)} "
            f"secuencias sanas; se requieren {n_background}."
        )
    bg_idx = rng.choice(len(healthy_train), size=n_background, replace=False)
    bg_X = np.stack([np.array(healthy_train[i]["X"]) for i in bg_idx], axis=0)
    background = torch.tensor(bg_X, dtype=torch.float32, device=device)

    # --- Muestra global: aleatorias del bloque de evaluación ---
    eval_size = len(sequences_eval)
    if eval_size < n_global_sample:
        raise ValueError(
            f"El bloque de evaluación tiene solo {eval_size} secuencias; "
            f"se requieren {n_global_sample}."
        )
    global_idx = rng.choice(eval_size, size=n_global_sample, replace=False)
    global_seqs = [sequences_eval[i] for i in global_idx]

    global_X      = np.stack([np.array(s["X"]) for s in global_seqs], axis=0)
    global_labels = np.array([s["failed"] for s in global_seqs])
    global_certs  = [s["CERT"] for s in global_seqs]
    global_periods = [s["period_end"] for s in global_seqs]
    global_inputs = torch.tensor(global_X, dtype=torch.float32, device=device)

    # --- Muestra local: positivos íntegros del bloque de evaluación ---
    positive_seqs = [s for s in sequences_eval if s["failed"] == 1]
    local_X      = np.stack([np.array(s["X"]) for s in positive_seqs], axis=0)
    local_labels = np.array([s["failed"] for s in positive_seqs])
    local_certs  = [s["CERT"] for s in positive_seqs]
    local_periods = [s["period_end"] for s in positive_seqs]
    local_inputs = torch.tensor(local_X, dtype=torch.float32, device=device)

    return {
        "background":     background,
        "global_inputs":  global_inputs,
        "global_labels":  global_labels,
        "global_certs":   global_certs,
        "global_periods": global_periods,
        "local_inputs":   local_inputs,
        "local_labels":   local_labels,
        "local_certs":    local_certs,
        "local_periods":  local_periods,
    }


# ---------------------------------------------------------------------------
# 1. Función núcleo: cálculo de SHAP values con GradientSHAP
# ---------------------------------------------------------------------------

def compute_gradient_shap(
    models: list,
    background: torch.Tensor,
    inputs: torch.Tensor,
    n_samples: int = 50,
    batch_size: int = 256,
    seed: int = 42,
) -> np.ndarray:
    """
    Calcula SHAP values mediante GradientSHAP para el ensamble.

    Aplica el axioma de aditividad: los SHAP values del ensamble son la media
    de los SHAP values de cada modelo individual.

    Los tensores se fuerzan a float32 internamente para garantizar compatibilidad
    con el LSTM independientemente del dtype de entrada.

    El modelo adaptado por el wrapper expone una salida (batch, 1), por lo que shap_values() 
    puede devolver una lista o un array con una dimensión final unitaria (N, 4, 192, 1). 
    El código incluye salvaguardas bi-direccionales para forzar siempre un array 
    plano de forma (N, 4, 192). Esto evita IndexError en modelos de salida escalar.

    Para el análisis global se recomienda pasar inputs en batches o usar
    una muestra representativa (~1000-2000 secuencias) para evitar OOM.
    Para el análisis local (33 secuencias positivas) se pasan íntegras.

    Args:
        models:     Lista de modelos LSTMBaseline en modo .eval().
        background: Tensor (n_bg, 4, 192) con el dataset de referencia.
        inputs:     Tensor (N, 4, 192) con las secuencias a explicar.
        n_samples:  Número de interpolaciones aleatorias por muestra.
                    Más muestras reducen varianza, aumentan tiempo de cómputo.
        batch_size: Tamaño de batch para procesar inputs en el análisis global.
                    Ignorado si N <= batch_size (análisis local típicamente).
        seed:       Semilla para reproducibilidad de las interpolaciones.

    Returns:
        shap_values: np.ndarray de forma (N, 4, 192).
                     Valores en espacio log-odds (escala del logit).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Forzar float32 para compatibilidad con LSTM
    background = background.float()
    inputs = inputs.float()

    all_shap_values = []

    for model in models:
        model.eval()
        # 1. Envolvemos el modelo individual con el wrapper bidimensional
        wrapped_model = LSTMBaselineSHAPWrapper(model)
        
        # 2. Le pasamos el modelo envuelto a SHAP
        explainer = shap.GradientExplainer(wrapped_model, background)

        n = inputs.shape[0]
        model_shap = []

        for start in range(0, n, batch_size):
            batch = inputs[start: start + batch_size]
            batch_shap = explainer.shap_values(batch, nsamples=n_samples)
            
            # Salvaguarda 1: Por si alguna versión de shap devuelve lista
            if isinstance(batch_shap, list):
                batch_shap = batch_shap[0]
                
            # Salvaguarda 2: contra la dimensión fantasma (N, 4, 192, 1) que añade
            # GradientExplainer cuando el wrapper expone una salida (batch, 1).
            if isinstance(batch_shap, np.ndarray) and batch_shap.ndim == 4 and batch_shap.shape[-1] == 1:
                batch_shap = batch_shap.squeeze(-1)
                
            model_shap.append(batch_shap)

        all_shap_values.append(np.concatenate(model_shap, axis=0))

    # Axioma de aditividad: media sobre los 3 modelos del ensamble
    shap_values = np.mean(np.stack(all_shap_values, axis=0), axis=0)

    return shap_values  # (N, 4, 192)


# ---------------------------------------------------------------------------
# 2. Agregación por timestep
# ---------------------------------------------------------------------------

def aggregate_shap_by_timestep(
    shap_values: np.ndarray,
    absolute: bool = True,
) -> np.ndarray:
    """
    Agrega SHAP values sobre las 192 dimensiones de cada timestep.

    Las 192 dimensiones son embeddings latentes de TabPFN sin interpretación
    regulatoria individual. La granularidad útil es el timestep (t-3, t-2,
    t-1, t), que representa la posición temporal de la ventana LSTM.

    Args:
        shap_values: np.ndarray de forma (N, 4, 192).
        absolute:    Si True, agrega |SHAP| → importancia neta por timestep.
                     Usar True para análisis global.
                     Si False, agrega SHAP con signo → preserva polaridad.
                     Usar False para análisis local: permite ver si un
                     trimestre empuja hacia quiebra (+) o retiene (-).

    Returns:
        np.ndarray de forma (N, 4). Una fila por secuencia, una columna
        por timestep en orden [t-3, t-2, t-1, t].
    """
    if absolute:
        return np.abs(shap_values).sum(axis=2)  # (N, 4)
    else:
        return shap_values.sum(axis=2)  # (N, 4)


# ---------------------------------------------------------------------------
# 3. Construcción del DataFrame de resultados
# ---------------------------------------------------------------------------

def build_shap_summary(
    shap_by_timestep: np.ndarray,
    labels: np.ndarray,
    period_ends: list,
    certs: list,
    absolute: bool = True,
) -> pd.DataFrame:
    """
    Construye el DataFrame base para análisis global y local.

    Análogo a df_preds en ranking_metrics: artefacto central del que
    parten todas las visualizaciones y análisis posteriores.

    Args:
        shap_by_timestep: np.ndarray (N, 4) de aggregate_shap_by_timestep.
        labels:           np.ndarray (N,) con etiquetas failed (0/1).
        period_ends:      Lista de N strings tipo "2022Q1".
        certs:            Lista de N identificadores CERT.
        absolute:         Documenta en el DataFrame si los valores son
                          absolutos (global) o con signo (local).

    Returns:
        DataFrame con columnas:
            CERT, period_end, failed,
            shap_t3, shap_t2, shap_t1, shap_t0,
            shap_mode ('absolute' | 'signed')
    """
    df = pd.DataFrame({
        "CERT":       certs,
        "period_end": period_ends,
        "failed":     labels,
        "shap_t3":    shap_by_timestep[:, 0],
        "shap_t2":    shap_by_timestep[:, 1],
        "shap_t1":    shap_by_timestep[:, 2],
        "shap_t0":    shap_by_timestep[:, 3],
        "shap_mode":  "absolute" if absolute else "signed",
    })
    return df


# ---------------------------------------------------------------------------
# 4. Visualización
# ---------------------------------------------------------------------------

def plot_shap_timestep_importance(
    shap_summary: pd.DataFrame,
    subset: str = "global",
    model_label: str = "LSTM Baseline",
    figsize: tuple = (8, 5),
    save_path: Optional[Path] = None,
) -> None:
    """
    Visualiza la importancia SHAP agregada por timestep.

    Para el análisis global (subset='global'): barplot de importancia media
    en valor absoluto sobre todas las secuencias del subset. Muestra la
    importancia relativa de cada posición temporal a nivel de población.

    Para el análisis local (subset='local'): una línea por CERT único con
    signo preservado. Permite ver la polaridad (contribución positiva hacia
    quiebra vs negativa de retención) y cómo evoluciona por banco.

    Args:
        shap_summary: DataFrame de build_shap_summary.
        subset:       'global' o 'local'. Controla el tipo de gráfico.
        model_label:  Etiqueta del modelo para título y leyenda.
        figsize:      Tamaño de la figura.
        save_path:    Si se provee, guarda la figura en esa ruta.
    """
    timestep_cols = ["shap_t3", "shap_t2", "shap_t1", "shap_t0"]
    timestep_labels = ["t-3", "t-2", "t-1", "t"]

    fig, ax = plt.subplots(figsize=figsize)

    if subset == "global":
        means = shap_summary[timestep_cols].mean()
        ax.bar(timestep_labels, means.values, color="steelblue", alpha=0.85)
        ax.set_ylabel("Importancia SHAP media |φ| (log-odds)")
        ax.set_title(
            f"Importancia SHAP por timestep — Análisis Global\n{model_label}"
        )
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))

    elif subset == "local":
        # Una línea por CERT único, con signo preservado
        for cert, group in shap_summary.groupby("CERT"):
            # Si hay varias filas por CERT (varias secuencias positivas),
            # usamos la media del CERT para la curva
            values = group[timestep_cols].mean().values
            ax.plot(timestep_labels, values, marker="o", label=str(cert), alpha=0.85)

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_ylabel("Contribución SHAP media φ (log-odds, con signo)")
        ax.set_title(
            f"Importancia SHAP por timestep — Análisis Local (secuencias positivas)\n{model_label}"
        )
        ax.legend(title="CERT", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)

    else:
        raise ValueError(f"subset debe ser 'global' o 'local', recibido: '{subset}'")

    ax.set_xlabel("Posición temporal en la ventana LSTM")
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figura guardada en: {save_path}")

    plt.show()