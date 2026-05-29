# src/models/tabular_encoder.py
"""
Wrapper de extracción de embeddings TabPFN para el pipeline de early warning bancario.
 
Responsabilidad única: dado un trimestre t y su contexto de trimestres anteriores,
produce e_tem_test(t) — la representación latente de cada banco en t condicionada
al contexto supervisado (X_train, y_train) mediante atención cruzada interna de TabPFN.
 
Uso en el pipeline
------------------
  Fase de desarrollo  → alimenta T-GCN como atributo de nodo X^t (bloque 2016Q2–2021Q4)
  Fase de evaluación  → ídem, con pesos de T-GCN/MLP/LSTM AE congelados (2022Q1–2025Q4)
 
Lo que este módulo NO hace
--------------------------
  - No extrae e_tem_train: existe dentro del forward pass de TabPFN pero es redundante
    (cada trimestre t ya fue procesado como X_test en su propia iteración) y no se
    persiste en ningún parquet.
  - No entrena ni actualiza pesos de TabPFN: actúa exclusivamente como encoder
    preentrenado con pesos fijos (Prior Labs, 2024).
  - No produce probabilidades de quiebra: ese rol corresponde al LSTM Baseline
    supervisado sobre CAMELS98 brutos.
 
Decisiones de diseño documentadas
----------------------------------
  VENTANA_CONTEXTO = 4 trimestres
    Coherente con el horizonte de predicción de Cole & Gunther (1995). Las señales
    de deterioro financiero son más informativas en los 4 trimestres previos a la
    quiebra que en historia más lejana. Mantiene X_train acotado (~24.000 filas),
    dentro de los límites de VRAM de la T4 de Google Colab.
 
  Leakage causal acotado en evaluación (documentado, no un error)
    A partir de 2023Q1, el contexto de 4 trimestres incluye trimestres del bloque
    de evaluación con sus etiquetas reales. Esto es tolerable: un regulador en
    2023Q1 sí conoce qué bancos quebraron en 2022. La tasa de positivos (~0.04%)
    hace que la influencia sobre el espacio latente de TabPFN sea mínima.
 
Referencias
-----------
  Cole, R. & Gunther, J. (1995) — horizonte 4 trimestres y ventana de contexto
  Prior Labs (2024) — TabPFN v2, in-context learning, extracción de embeddings
"""
 
import gc
import time
 
import numpy as np
import pandas as pd
import torch
 
# =============================================================================
# Constantes
# =============================================================================
 
VENTANA_CONTEXTO: int = 4  # trimestres de contexto para TabPFN
 
 
# =============================================================================
# Función principal de extracción
# =============================================================================
 
def extraer_embeddings_trimestre(
    panel: pd.DataFrame,
    feature_cols: list[str],
    periodo_test: str,
    periodos_train: list[str],
    embedding_extractor,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Extrae e_tem_test de TabPFN para un trimestre t usando ventana deslizante.
 
    TabPFN actúa como encoder preentrenado con pesos fijos. No hay backprop
    ni actualización de pesos — solo forward pass para obtener la representación
    latente de X_test condicionada al contexto supervisado (X_train, y_train)
    por atención cruzada interna.
 
    El embedding resultante captura el estado financiero del banco en t
    reinterpretado a través de la semántica de riesgo del contexto: un banco
    con ratios similares a bancos que quebraron en el contexto quedará cerca
    de ellos en el espacio latente.
 
    Parameters
    ----------
    panel              : DataFrame completo con features, etiquetas e IDs.
                         Debe contener columnas 'period', 'CERT', 'failed'.
    feature_cols       : Lista de columnas de features (excluye 'CERT', 'period', 'failed').
    periodo_test       : Trimestre t a embeddear (e.g. '2017Q1').
    periodos_train     : Lista de todos los trimestres anteriores a t.
                         Se recorta internamente a VENTANA_CONTEXTO — el argumento
                         original del caller no se muta.
    embedding_extractor: Instancia de EmbeddingExtractor de TabPFN ya inicializada.
 
    Returns
    -------
    e_test    : np.ndarray de forma (n_bancos_t, d_emb)
                Embedding de cada banco en t. Alimenta T-GCN como atributo de nodo X^t.
    meta_test : DataFrame con columnas ['CERT', 'period'] del trimestre t.
                Permite reconstruir la identidad de cada fila de e_test.
 
    Notes
    -----
    Solo se realiza una llamada a get_embeddings() con data_source='test'.
    La llamada con data_source='train' es innecesaria: e_tem_train existe dentro
    del transformer durante el forward pass pero no se necesita fuera de TabPFN.
    """
    # Ventana deslizante — variable local para no mutar el argumento del caller
    ctx_periodos = periodos_train[-VENTANA_CONTEXTO:]
 
    mask_train = panel['period'].isin(ctx_periodos)
    mask_test  = panel['period'] == periodo_test
 
    X_train = panel[mask_train][feature_cols].values
    y_train = panel[mask_train]['failed'].values
    X_test  = panel[mask_test][feature_cols].values
 
    meta_test = panel[mask_test][['CERT', 'period']].reset_index(drop=True)
 
    print(
        f"| ctx: {ctx_periodos[0]}→{ctx_periodos[-1]} ({len(ctx_periodos)}Q) "
        f"X_train: {X_train.shape} X_test: {X_test.shape} ",
        end=''
    )
 
    # Una sola llamada — solo e_tem_test es necesario fuera de TabPFN
    test_emb = embedding_extractor.get_embeddings(
        X_train, y_train, X_test, data_source='test'
    )
    e_test = test_emb[0]
 
    # Liberar memoria GPU entre trimestres
    gc.collect()
    torch.cuda.empty_cache()
 
    return e_test, meta_test
 
 
# =============================================================================
# Loop de extracción sobre un bloque temporal completo
# =============================================================================
 
def extraer_embeddings_bloque(
    panel: pd.DataFrame,
    feature_cols: list[str],
    periodos_bloque: list[str],
    embedding_extractor,
    ruta_salida,
    prefijo: str = 'emb',
) -> pd.DataFrame:
    """
    Extrae embeddings TabPFN para todos los trimestres de un bloque temporal.
 
    Diseñado para ser reutilizado tanto en la fase de desarrollo (2016Q2–2021Q4)
    como en la fase de evaluación (2022Q1–2025Q4) sin cambios de interfaz.
 
    El primer trimestre del bloque se omite automáticamente por carecer de
    contexto previo dentro del propio bloque. En la fase de evaluación, el
    contexto del primer trimestre proviene de los últimos trimestres del bloque
    de desarrollo — esto debe gestionarse externamente pasando periodos_bloque
    con los trimestres de desarrollo prepended si se desea ese comportamiento.
 
    Parameters
    ----------
    panel              : DataFrame completo con features, etiquetas e IDs.
    feature_cols       : Columnas de features.
    periodos_bloque    : Lista ordenada de trimestres del bloque a procesar.
    embedding_extractor: Instancia de EmbeddingExtractor de TabPFN.
    ruta_salida        : Path (pathlib.Path) al directorio donde guardar
                         checkpoints trimestrales y el parquet consolidado.
    prefijo            : Prefijo para los nombres de archivo de checkpoint
                         (default: 'emb').
 
    Returns
    -------
    DataFrame consolidado con columnas ['CERT', 'period', 'emb_0', ..., 'emb_N'].
    También escribe en disco:
      - {ruta_salida}/{prefijo}_{periodo}.parquet  — checkpoint por trimestre
      - {ruta_salida}/{prefijo}_consolidado.parquet — consolidado final
 
    Notes
    -----
    Los checkpoints trimestrales protegen contra desconexiones de Colab:
    si el loop se interrumpe, los trimestres ya procesados no se pierden.
    """
    def _fmt_tiempo(segundos: float) -> str:
        """Formatea segundos como mm:ss o hh:mm:ss según magnitud."""
        h = int(segundos // 3600)
        m = int((segundos % 3600) // 60)
        s = int(segundos % 60)
        return f"{h:02d}h{m:02d}m{s:02d}s" if h > 0 else f"{m:02d}m{s:02d}s"
 
    nombre_bloque = f"{periodos_bloque[0]}–{periodos_bloque[-1]}"
    print(f"=== EXTRACCIÓN {nombre_bloque} ===\n")
 
    embeddings_bloque = []
    tiempos_trimestre = []
    inicio_total = time.perf_counter()
 
    for i, t in enumerate(periodos_bloque):
 
        if i == 0:
            print(f"[{t}] Omitido — sin contexto previo\n")
            continue
 
        periodos_ctx = periodos_bloque[:i]  # todos los anteriores a t
                                             # la función recorta a VENTANA_CONTEXTO
        inicio_t = time.perf_counter()
        print(f"[{t}] ", end='')
 
        try:
            e_test, meta_test = extraer_embeddings_trimestre(
                panel, feature_cols, t, periodos_ctx, embedding_extractor
            )
 
            dim      = e_test.shape[1]
            emb_cols = [f'emb_{j}' for j in range(dim)]
 
            df_t = pd.DataFrame(e_test, columns=emb_cols)
            df_t.insert(0, 'CERT',   meta_test['CERT'].values)
            df_t.insert(1, 'period', meta_test['period'].values)
 
            embeddings_bloque.append(df_t)
 
            # Checkpoint trimestral — protege contra desconexiones de Colab
            df_t.to_parquet(ruta_salida / f'{prefijo}_{t}.parquet', index=False)
 
            elapsed_t = time.perf_counter() - inicio_t
            tiempos_trimestre.append(elapsed_t)
 
            # ETA: media móvil excluyendo el primer trimestre (descarga checkpoint HF)
            tiempos_para_eta = tiempos_trimestre[1:] if len(tiempos_trimestre) > 1 else tiempos_trimestre
            trimestres_restantes = len(periodos_bloque) - 1 - i
            eta = np.mean(tiempos_para_eta) * trimestres_restantes
 
            print(f"| shape: {e_test.shape} | {_fmt_tiempo(elapsed_t)} | ETA: {_fmt_tiempo(eta)} ✓\n")
 
        except Exception as ex:
            elapsed_t = time.perf_counter() - inicio_t
            tiempos_trimestre.append(elapsed_t)
            print(f"| ERROR: {ex} | {_fmt_tiempo(elapsed_t)}\n")
 
    # --- Consolidar en un único parquet ---
    if embeddings_bloque:
        emb_df = pd.concat(embeddings_bloque, ignore_index=True)
        emb_df.to_parquet(ruta_salida / f'{prefijo}_consolidado.parquet', index=False)
        print(f"\nEmbeddings consolidados: {emb_df.shape}")
    else:
        emb_df = pd.DataFrame()
        print("\nNo se generaron embeddings — revisar errores anteriores.")
 
    # --- Resumen de tiempos ---
    tiempo_total = time.perf_counter() - inicio_total
    print(f"\n{'='*50}")
    print(f"Tiempo total          : {_fmt_tiempo(tiempo_total)}")
    if tiempos_trimestre:
        print(f"Media por trimestre   : {_fmt_tiempo(float(np.mean(tiempos_trimestre)))}")
        print(f"Trimestre más lento   : {_fmt_tiempo(max(tiempos_trimestre))}")
        print(f"Trimestre más rápido  : {_fmt_tiempo(min(tiempos_trimestre))}")
    print(f"{'='*50}")
 
    return emb_df
 
 
# =============================================================================
# Verificación del bloque extraído
# =============================================================================
 
def verificar_embeddings_bloque(
    ruta_parquet,
    periodos_esperados: list[str],
    periodos_contaminantes: list[str] | None = None,
) -> bool:
    """
    Verifica integridad del parquet de embeddings de un bloque temporal.
 
    Checks realizados:
      1. Rango temporal y número de trimestres coinciden con periodos_esperados
      2. Dimensión de embeddings consistente
      3. Ausencia de NaNs
      4. Separación temporal: ningún periodo contaminante presente
 
    Parameters
    ----------
    ruta_parquet           : Path al parquet consolidado a verificar.
    periodos_esperados     : Lista de trimestres que deben estar presentes
                             (excluye el primer trimestre omitido).
    periodos_contaminantes : Lista de periodos que NO deben aparecer en el bloque
                             (e.g., PERIODOS_EVALUACION al verificar desarrollo).
                             Si None, el check de separación temporal se omite.
 
    Returns
    -------
    bool : True si todos los checks pasan, False si alguno falla.
    """
    emb = pd.read_parquet(ruta_parquet)
    emb_cols = [c for c in emb.columns if c.startswith('emb_')]
    ok = True
 
    print(f"=== VERIFICACIÓN {ruta_parquet.name} ===\n")
    print(f"Shape total          : {emb.shape}")
    print(f"Trimestres           : {emb['period'].nunique()}  (esperados: {len(periodos_esperados)})")
    print(f"Rango temporal       : {emb['period'].min()} → {emb['period'].max()}")
    print(f"Bancos únicos        : {emb['CERT'].nunique()}")
    print(f"Dimensión embeddings : {len(emb_cols)}")
 
    # NaNs
    n_nans = emb[emb_cols].isna().sum().sum()
    estado_nans = '✓' if n_nans == 0 else '⚠ REVISAR'
    print(f"NaNs en embeddings   : {n_nans}  {estado_nans}")
    if n_nans > 0:
        ok = False
 
    # Trimestres faltantes
    trimestres_presentes = sorted(emb['period'].unique())
    faltantes = set(periodos_esperados) - set(trimestres_presentes)
    print(f"Trimestres faltantes : {faltantes if faltantes else 'ninguno ✓'}")
    if faltantes:
        ok = False
 
    # Separación temporal
    if periodos_contaminantes is not None:
        cruce = set(trimestres_presentes) & set(periodos_contaminantes)
        if cruce:
            print(f"\n⚠ LEAKAGE DETECTADO — periodos contaminantes presentes: {cruce}")
            ok = False
        else:
            print(f"\nSanity check temporal : ningún periodo contaminante ✓")
 
    # Distribución de bancos por trimestre
    por_trimestre = emb.groupby('period').size()
    print(f"\nBancos por trimestre  : media={por_trimestre.mean():.0f} "
          f"mín={por_trimestre.min()} máx={por_trimestre.max()}")
 
    print(f"\n{'La verificación es correcta ✓' if ok else '⚠ Verificación fallida — revisar errores'}")
    return ok