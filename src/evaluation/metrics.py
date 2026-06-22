# src/evaluation/metrics

"""
Funciones de evaluación agnósticas al
modelo concreto (reciben model como argumento). Reutilizables sin cambios
para el Baseline y para el futuro Híbrido, ya que ambos comparten la
misma arquitectura y el mismo protocolo de entrenamiento.

Métricas de ranking trimestral para apoyo a supervisión.
 
Implementa Hit-Rate@K Trimestral: capacidad del modelo para concentrar
las quiebras reales dentro de las K entidades de mayor riesgo predicho
en cada trimestre, evaluado en aislamiento (el ranking de un trimestre
nunca se mezcla con el universo de otro trimestre).
 
Diseñado para ser independiente de qué modelo generó las predicciones:
toma como entrada el dict que ya devuelve evaluate_ensemble_full (o
cualquier dict con la misma forma: probs, targets, cert, period_end),
de modo que la misma función se reutiliza sin cambios para el LSTM
baseline y para el modelo Híbrido, asegurando que la comparación entre
ambos use idéntica definición de métrica.
"""


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, precision_recall_curve
import matplotlib.pyplot as plt

import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Utilidades de aritmética de trimestres
# ---------------------------------------------------------------------------
 
def _parse_period(period_str: str) -> tuple[int, int]:
    """'2022Q1' → (2022, 1)"""
    year, q = period_str.split("Q")
    return int(year), int(q)
 
 
def _period_to_str(year: int, q: int) -> str:
    return f"{year}Q{q}"
 
 
def _subtract_quarters(period_str: str, n: int) -> str:
    """
    Resta n trimestres a un periodo dado.
    Ejemplo: '2022Q1' - 1 → '2021Q4', '2022Q3' - 2 → '2022Q1'.
    """
    year, q = _parse_period(period_str)
    total = year * 4 + (q - 1) - n
    return _period_to_str(total // 4, total % 4 + 1)
 
 
# ---------------------------------------------------------------------------


@torch.no_grad()
def ensemble_predict(models: list, x: torch.Tensor) -> torch.Tensor:
    """
    Promedio aritmético de PROBABILIDADES, no de logits: sigmoid no es
    lineal, así que promediar logits y aplicar sigmoid después no da el
    mismo resultado que aplicar sigmoid por modelo y promediar después,
    sobre todo cerca de la saturación. La fórmula que escribiste ya
    especifica el orden correcto (sigma(logits) primero, promedio
    después); esto es solo la implementación fiel de esa fórmula.
    """
    probs = torch.stack([torch.sigmoid(m(x)) for m in models], dim=0)
    return probs.mean(dim=0)


@torch.no_grad()
def evaluate_ensemble(models: list, loader, device: str = "cpu"):
    """
    AUC-PR/AUC-ROC calculado sobre las probabilidades YA promediadas del
    ensamble, no sobre el promedio de los AUC-PR individuales de cada
    modelo. Esta es la métrica que caracteriza al LSTMBaseline final,
    distinta de los tres escalares por semilla reportados hasta ahora.
    """
    all_probs, all_targets = [], []
    for batch in loader:
        x = batch["X"].to(device)
        y = batch["failed"]
        probs = ensemble_predict(models, x).cpu()
        all_probs.append(probs)
        all_targets.append(y)

    probs = torch.cat(all_probs).numpy()
    targets = torch.cat(all_targets).numpy()

    auc_pr = average_precision_score(targets, probs)
    auc_roc = roc_auc_score(targets, probs)
    return auc_pr, auc_roc

@torch.no_grad()
def evaluate_ensemble_full(models: list, loader, device: str = "cpu"):
    """
    Extensión de evaluate_ensemble para la fase de evaluación final.
    Devuelve probabilidades y metadatos crudos (CERT, period_end), no
    solo las métricas agregadas, porque Brier Score y F1-Score Max
    requieren el array completo de probabilidades, y la trazabilidad
    por banco/trimestre se necesita para análisis posteriores. No
    sustituye a evaluate_ensemble, que sigue usándose tal cual donde ya
    está integrada (búsqueda de hiperparámetros, selección de época).
    """
    all_probs, all_targets, all_cert, all_period = [], [], [], []

    for batch in loader:
        x = batch["X"].to(device)
        y = batch["failed"]
        probs = ensemble_predict(models, x).cpu()

        all_probs.append(probs)
        all_targets.append(y)
        all_cert.extend(batch["CERT"])
        all_period.extend(batch["period_end"])

    probs = torch.cat(all_probs).numpy()
    targets = torch.cat(all_targets).numpy()

    auc_pr = average_precision_score(targets, probs)
    auc_roc = roc_auc_score(targets, probs)
    brier = brier_score_loss(targets, probs)

    precision, recall, thresholds = precision_recall_curve(targets, probs)
    p, r = precision[:-1], recall[:-1]  # último punto no tiene threshold asociado
    denom = p + r
    f1_scores = np.where(denom > 0, 2 * p * r / np.where(denom == 0, 1, denom), 0.0)
    f1_max_idx = f1_scores.argmax()
    f1_max = f1_scores[f1_max_idx]
    f1_max_threshold = thresholds[f1_max_idx]

    return {
        "auc_pr": auc_pr,
        "auc_roc": auc_roc,
        "brier_score": brier,
        "f1_max": f1_max,
        "f1_max_threshold": f1_max_threshold,
        "probs": probs,
        "targets": targets,
        "cert": all_cert,
        "period_end": all_period,
    }



def build_eval_predictions_df(resultados_eval: dict) -> pd.DataFrame:
    """
    Construye el dataframe base de predicciones por muestra a partir del
    dict devuelto por evaluate_ensemble_full. Es el único punto de
    entrada de datos crudos para las métricas de ranking trimestral, de
    forma que Hit-Rate@K y la futura Curva de Anticipación Temporal
    parten exactamente del mismo artefacto y no hay riesgo de
    inconsistencia entre ambos análisis.
    """
    required_keys = {"cert", "period_end", "targets", "probs"}
    missing = required_keys - set(resultados_eval.keys())
    assert not missing, f"resultados_eval no contiene las claves esperadas: {missing}"
 
    df = pd.DataFrame({
        "CERT": resultados_eval["cert"],
        "period_end": resultados_eval["period_end"],
        "failed": np.asarray(resultados_eval["targets"]).astype(int),
        "prob": np.asarray(resultados_eval["probs"]),
    })
    return df


def build_eval_predictions_df(resultados_eval: dict) -> pd.DataFrame:
    """
    Construye el dataframe base de predicciones por muestra a partir del
    dict devuelto por evaluate_ensemble_full. Es el único punto de
    entrada de datos crudos para las métricas de ranking trimestral, de
    forma que Hit-Rate@K y la futura Curva de Anticipación Temporal
    parten exactamente del mismo artefacto y no hay riesgo de
    inconsistencia entre ambos análisis.
    """
    required_keys = {"cert", "period_end", "targets", "probs"}
    missing = required_keys - set(resultados_eval.keys())
    assert not missing, f"resultados_eval no contiene las claves esperadas: {missing}"
 
    df = pd.DataFrame({
        "CERT": resultados_eval["cert"],
        "period_end": resultados_eval["period_end"],
        "failed": np.asarray(resultados_eval["targets"]).astype(int),
        "prob": np.asarray(resultados_eval["probs"]),
    })
    return df
 
 
 
 
def hit_rate_at_k_trimestral(
    df_preds: pd.DataFrame,
    k_values: list[int],
    verbose: bool = True,
) -> dict:
    """
    Calcula Hit-Rate@K trimestral para cada K en k_values.
 
    Para cada trimestre (period_end) se ordenan las entidades activas
    ese trimestre por probabilidad de quiebra predicha, de mayor a
    menor, y se toman las K de mayor riesgo (Top-K(t)). Hit-Rate@K(t) es
    la fracción de quiebras reales de ese trimestre que caen dentro de
    Top-K(t): |P(t) ∩ TopK(t)| / |P(t)|. El ranking se hace en
    aislamiento, solo contra las demás entidades activas en ese mismo
    trimestre, nunca contra el pool global de todos los trimestres de
    evaluación.
 
    Los trimestres sin ninguna quiebra real no participan en el
    agregado (el ratio quedaría 0/0, indefinido), pero se devuelven
    aparte junto con la composición de su Top-K, como lista de
    vigilancia sin verificación posible ese trimestre concreto.
 
    El agregado final es un macro-promedio sobre los trimestres con al
    menos una quiebra real: cada trimestre pesa igual en el promedio
    con independencia de cuántas quiebras tuviera, en línea con
    evaluar "cada trimestre en aislamiento" en vez de darle más peso a
    los trimestres con más eventos.
 
    Nota sobre empates: en caso de probabilidades idénticas en la
    frontera del Top-K, pandas conserva el orden original de aparición
    (sort estable). Con salidas continuas de sigmoide es extremadamente
    improbable que esto afecte al resultado, pero si se necesitara un
    criterio de desempate explícito y reproducible (p. ej. por CERT),
    se puede añadir como columna de desempate antes del sort_values.
 
    Devuelve un dict indexado por K, cada uno con:
      - "macro_avg": Hit-Rate@K agregado (macro-promedio)
      - "per_period": tabla con hit_rate, n_failed y n_hits por trimestre
      - "watchlist_zero_failure_quarters": composición del Top-K en los
        trimestres sin quiebras reales, para inspección cualitativa
    """
    assert len(k_values) > 0, "k_values no puede estar vacío."
    assert all(isinstance(k, int) and k > 0 for k in k_values), "k_values debe ser una lista de enteros positivos."
 
    results = {}
 
    for k in k_values:
        per_period_rows = []
        watchlist_rows = []
 
        for period, group in df_preds.groupby("period_end", sort=True):
            n_active = len(group)
            n_failed = int(group["failed"].sum())
            top_k = group.sort_values("prob", ascending=False).head(k)
 
            if n_failed == 0:
                watchlist_rows.append({
                    "period_end": period,
                    "n_active": n_active,
                    "top_k_certs": top_k["CERT"].tolist(),
                    "top_k_probs": top_k["prob"].round(4).tolist(),
                })
                continue
 
            n_hits = int(top_k["failed"].sum())
            per_period_rows.append({
                "period_end": period,
                "n_active": n_active,
                "n_failed": n_failed,
                "n_hits": n_hits,
                "hit_rate": n_hits / n_failed,
            })
 
        df_per_period = pd.DataFrame(per_period_rows)
        df_watchlist = pd.DataFrame(watchlist_rows)
        macro_avg = df_per_period["hit_rate"].mean() if len(df_per_period) else float("nan")
 
        results[k] = {
            "macro_avg": macro_avg,
            "per_period": df_per_period,
            "watchlist_zero_failure_quarters": df_watchlist,
        }
 
        if verbose:
            print("\n" + "=" * 60)
            print(f"HIT-RATE@{k} TRIMESTRAL")
            print("=" * 60)
            print(f"Trimestres con quiebras reales  : {len(df_per_period)}")
            print(f"Trimestres sin quiebras reales   : {len(df_watchlist)}  (excluidos del macro-promedio)")
            print(f"Hit-Rate@{k} (macro-promedio)    : {macro_avg:.4f}")
            if len(df_per_period):
                print(df_per_period.to_string(index=False))
            print("=" * 60)
 
    return results


def rank_quiebras_reales(df_preds: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Calcula, para cada quiebra real (failed=1) del bloque de evaluación,
    su posición de rank dentro del ranking de riesgo de su propio
    trimestre (rank=1 es la entidad de mayor probabilidad predicha ese
    trimestre), en aislamiento respecto a los demás trimestres, con el
    mismo criterio de ordenación y desempate que hit_rate_at_k_trimestral
    (sort estable de pandas), para que ambas funciones sean consistentes
    entre sí: una quiebra con rank=5 debe figurar como acierto en
    hit_rate_at_k_trimestral para K=5, y así sucesivamente.
 
    Es el diagnóstico complementario a Hit-Rate@K: una quiebra que no
    entra en el top-K puede estar rankeada justo fuera de él
    (casi-acierto, recuperable ampliando K) o muy abajo en el ranking
    (fallo profundo, no recuperable ampliando K dentro de rangos
    razonables, indicio de que el modelo no generó señal de riesgo
    alguna para esa entidad ese trimestre). Distinguir entre ambos casos
    es necesario antes de decidir si ampliar el margen de K tiene
    sentido o si el problema es de otra naturaleza.
 
    Devuelve un dataframe con una fila por cada quiebra real (33 filas
    en el bloque de evaluación completo), con CERT, period_end, prob,
    rank (1-indexado), n_active, rank_pct (rank/n_active, percentil de
    riesgo dentro del trimestre, 0=máximo riesgo) y los indicadores
    booleanos hit_top5 / hit_top10 para cruce directo con la tabla de
    Hit-Rate@K.
    """
    rows = []
 
    for period, group in df_preds.groupby("period_end", sort=True):
        n_active = len(group)
        ranked = group.sort_values("prob", ascending=False).reset_index(drop=True)
        ranked["rank"] = np.arange(1, n_active + 1)
 
        failures = ranked[ranked["failed"] == 1]
        for _, row in failures.iterrows():
            rows.append({
                "CERT": row["CERT"],
                "period_end": period,
                "prob": row["prob"],
                "rank": int(row["rank"]),
                "n_active": n_active,
                "rank_pct": row["rank"] / n_active,
                "hit_top5": bool(row["rank"] <= 5),
                "hit_top10": bool(row["rank"] <= 10),
            })
 
    df_rank = pd.DataFrame(rows).sort_values(["period_end", "rank"]).reset_index(drop=True)
 
    if verbose:
        print("\n" + "=" * 70)
        print("RANK DE QUIEBRAS REALES DENTRO DE SU TRIMESTRE (EN AISLAMIENTO)")
        print("=" * 70)
        print(f"Total de quiebras reales en evaluación : {len(df_rank)}")
        print(df_rank.to_string(index=False))
        print("=" * 70)
 
    return df_rank


def build_anticipation_trajectories(
    df_preds: pd.DataFrame,
    window: int = 4,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Construye las trayectorias longitudinales de probabilidad predicha
    para cada entidad única con al menos una quiebra real (failed=1) en
    el bloque de evaluación, en los `window` trimestres que preceden
    (e incluyen) su último trimestre de alerta.
 
    Criterio de anclaje temporal: se usa el ÚLTIMO trimestre en que el
    CERT aparece como failed=1 como punto t (resolución definitiva del
    proceso de deterioro), y se retrocede hasta t-(window-1). Esto es
    coherente con el marco regulatorio en que la bandera failed se
    propaga durante varios trimestres consecutivos hasta la intervención
    definitiva; anclar en el último trimestre garantiza que t representa
    el estado de máximo deterioro observable en los datos, no una
    detección prematura.
 
    Para cada punto de la trayectoria (t, t-1, t-2, t-3) se busca la
    probabilidad predicha por el ensamble para ese CERT en ese periodo
    concreto dentro de df_preds. Si el CERT no tiene secuencia en algún
    periodo de la ventana (por ejemplo porque en ese trimestre el banco
    aún no estaba activo en el panel, o porque el periodo cae fuera del
    bloque de evaluación), el punto queda como NaN — no se imputa, para
    no falsear la trayectoria.
 
    Nota metodológica sobre la etiqueta failed: la bandera failed=1 en
    la base de la FDIC es una alerta regulatoria propagada, no una
    certificación puntual de quiebra. Que un CERT aparezca como
    failed=1 en múltiples trimestres consecutivos refleja el proceso
    administrativo de deterioro y resolución, no necesariamente que el
    banco quebró en cada uno de esos trimestres. La confirmación de si
    la resolución final fue quiebra, fusión o absorción solo es
    verificable en el modelo Híbrido, comprobando si el nodo del banco
    desaparece del grafo relacional tras t. Esta limitación debe
    declararse en la sección de limitaciones del TFM.
 
    Devuelve un dataframe con columnas:
      CERT, period_anchor (t), t_minus_0..t_minus_{window-1} (periodos
      absolutos), prob_t, prob_t1, prob_t2, prob_t3 (probabilidades),
      n_quarters_available (cuántos puntos de la trayectoria tienen prob
      no nula, útil para filtrar trayectorias incompletas en el gráfico).
    """
    lookup = df_preds.set_index(["CERT", "period_end"])["prob"]
 
    failing_certs = df_preds[df_preds["failed"] == 1]["CERT"].unique()
    rows = []
 
    for cert in failing_certs:
        cert_failures = df_preds[
            (df_preds["CERT"] == cert) & (df_preds["failed"] == 1)
        ]["period_end"].tolist()
 
        # Ancla en el último trimestre de alerta
        cert_failures_sorted = sorted(cert_failures, key=lambda p: _parse_period(p))
        t_anchor = cert_failures_sorted[-1]
 
        row = {"CERT": cert, "period_anchor": t_anchor}
        probs = []
        for step in range(window):
            period_label = f"t_minus_{step}"
            prob_label = f"prob_t{step}"
            period = _subtract_quarters(t_anchor, step)
            row[period_label] = period
            prob = lookup.get((cert, period), np.nan)
            row[prob_label] = prob
            probs.append(prob)
 
        row["n_quarters_available"] = int(sum(p == p for p in probs))  # cuenta no-NaN
        rows.append(row)
 
    df_traj = (
        pd.DataFrame(rows)
        .sort_values("period_anchor")
        .reset_index(drop=True)
    )
 
    if verbose:
        print("\n" + "=" * 60)
        print("TRAYECTORIAS DE ANTICIPACIÓN — RESUMEN")
        print("=" * 60)
        print(f"Entidades únicas con alerta failed=1 : {len(df_traj)}")
        print(f"Ventana temporal                     : {window} trimestres")
        print(f"Trayectorias completas (sin NaN)     : {(df_traj['n_quarters_available'] == window).sum()}")
        cols_show = ["CERT", "period_anchor"] + [f"prob_t{i}" for i in range(window)]
        print(df_traj[cols_show].to_string(index=False))
        print("=" * 60)
 
    return df_traj
 
 
def plot_anticipation_curve(
    df_traj: pd.DataFrame,
    window: int = 4,
    model_label: str = "LSTM Baseline",
    figsize: tuple = (12, 6),
    save_path: str | None = None,
) -> None:
    """
    Gráfico de la Curva de Anticipación Temporal: una línea por entidad
    única, mostrando la evolución de la probabilidad predicha desde
    t-(window-1) hasta t (trimestre de resolución definitiva del proceso
    de deterioro), con t en el eje derecho.
 
    El eje X representa el tiempo relativo al colapso: t-3, t-2, t-1, t.
    Cada línea corresponde a un CERT único; el color distingue las
    entidades. Las líneas con puntos NaN se trazan con los puntos
    disponibles y se marcan con marcadores huecos para dejar constancia
    visual de la trayectoria incompleta (sin imputar valores faltantes).
 
    El gráfico se diseña como herramienta de diagnóstico para el
    regulador: la pendiente de cada curva indica la velocidad de
    aceleración del riesgo, y la posición de cada curva en t refleja
    si el modelo llegó a detectar correctamente la entidad en el momento
    de la resolución.
 
    Parámetros
    ----------
    df_traj     : salida de build_anticipation_trajectories
    window      : mismo valor usado al construir df_traj (default 4)
    model_label : etiqueta para el título (permite reutilizar en Híbrido)
    figsize     : tamaño de la figura
    save_path   : si se proporciona, guarda la figura en esa ruta
    """
    prob_cols = [f"prob_t{i}" for i in range(window - 1, -1, -1)]  # t-3 → t
    x_labels = [f"t-{i}" if i > 0 else "t" for i in range(window - 1, -1, -1)]
    x = np.arange(window)
 
    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.get_cmap("tab10")
 
    for idx, (_, row) in enumerate(df_traj.iterrows()):
        probs = [row[col] for col in prob_cols]
        color = cmap(idx % 10)
        cert_label = f"CERT {int(row['CERT'])} (t={row['period_anchor']})"
 
        # Trazar solo los puntos no nulos, conectados
        valid_x = [xi for xi, p in zip(x, probs) if p == p]
        valid_p = [p for p in probs if p == p]
 
        ax.plot(valid_x, valid_p, marker="o", color=color,
                linewidth=1.8, markersize=5, label=cert_label)
 
        # Marcar puntos NaN con marcador hueco para indicar ausencia de dato
        missing_x = [xi for xi, p in zip(x, probs) if p != p]
        if missing_x:
            ax.scatter(missing_x, [0] * len(missing_x), marker="x",
                       color=color, s=40, zorder=5)
 
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=11)
    ax.set_xlabel("Trimestres previos al colapso", fontsize=12)
    ax.set_ylabel("Probabilidad de quiebra predicha", fontsize=12)
    ax.set_title(
        f"Curva de Anticipación Temporal — {model_label}\n"
        f"Evolución de P(quiebra) en los {window} trimestres previos al colapso",
        fontsize=13,
    )
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
    ax.set_ylim(-0.05, 1.05)
    ax.axvline(x=window - 1, color="gray", linestyle="--", linewidth=1,
               alpha=0.6, label="t: resolución definitiva")
    ax.legend(fontsize=8, loc="upper left", ncol=2, framealpha=0.7)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
 
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figura guardada en: {save_path}")
 
    plt.show()


def plot_full_trajectories(
    df_preds: pd.DataFrame,
    model_label: str = "LSTM Baseline",
    figsize: tuple = (14, 6),
    save_path: str | None = None,
) -> None:
    """
    Trayectoria longitudinal completa en tiempo absoluto para cada entidad
    única con al menos una quiebra real (failed=1) en el bloque de
    evaluación. A diferencia de plot_anticipation_curve, que normaliza el
    eje temporal al momento del colapso (t-3, t-2, t-1, t), aquí el eje X
    son los trimestres reales del bloque de evaluación (2022Q1 → 2025Q4),
    lo que permite leer directamente en qué momento del calendario el
    modelo empezó a generar señal para cada entidad.
 
    Para cada CERT se trazan todos los trimestres en que aparece en
    df_preds, no solo la ventana de 4 pasos previa al colapso. Los
    periodos en que la etiqueta failed=1 se marcan con un punto más
    grande y relleno para distinguirlos visualmente de los trimestres
    donde el banco estaba activo pero sin alerta.
 
    Esta visualización es complementaria a plot_anticipation_curve:
    - plot_anticipation_curve: ¿cuántos trimestres antes despega la señal?
      (tiempo relativo, comparable entre bancos con distintas fechas de colapso)
    - plot_full_trajectories: ¿cuándo exactamente en el calendario?
      (tiempo absoluto, directamente legible por un supervisor regulatorio)
    """
    failing_certs = df_preds[df_preds["failed"] == 1]["CERT"].unique()
 
    # Orden cronológico de todos los periodos presentes en el bloque
    all_periods = sorted(df_preds["period_end"].unique(), key=_parse_period)
    period_to_x = {p: i for i, p in enumerate(all_periods)}
    n_periods = len(all_periods)
 
    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.get_cmap("tab10")
 
    for idx, cert in enumerate(sorted(failing_certs)):
        cert_df = (
            df_preds[df_preds["CERT"] == cert]
            .sort_values("period_end", key=lambda s: s.map(_parse_period))
            .reset_index(drop=True)
        )
        color = cmap(idx % 10)
 
        xs = [period_to_x[p] for p in cert_df["period_end"]]
        ys = cert_df["prob"].tolist()
 
        ax.plot(xs, ys, color=color, linewidth=1.6,
                label=f"CERT {int(cert)}", zorder=2)
 
        # Trimestres con alerta failed=1: marcador grande relleno
        failed_mask = cert_df["failed"] == 1
        xs_failed = [period_to_x[p] for p in cert_df.loc[failed_mask, "period_end"]]
        ys_failed = cert_df.loc[failed_mask, "prob"].tolist()
        ax.scatter(xs_failed, ys_failed, color=color, s=60,
                   zorder=4, marker="o")
 
        # Trimestres sin alerta: marcador pequeño hueco
        xs_ok = [period_to_x[p] for p in cert_df.loc[~failed_mask, "period_end"]]
        ys_ok = cert_df.loc[~failed_mask, "prob"].tolist()
        ax.scatter(xs_ok, ys_ok, color=color, s=15, zorder=3,
                   marker="o", facecolors="none", linewidths=0.8)
 
    ax.set_xticks(range(n_periods))
    ax.set_xticklabels(all_periods, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Probabilidad de quiebra predicha", fontsize=12)
    ax.set_xlabel("Trimestre (tiempo absoluto)", fontsize=12)
    ax.set_title(
        f"Trayectorias Longitudinales Completas — {model_label}\n"
        "Evolución de P(quiebra) a lo largo del bloque de evaluación "
        "(● = trimestres con alerta failed=1)",
        fontsize=12,
    )
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, loc="upper left", ncol=2, framealpha=0.7)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.grid(axis="x", linestyle=":", alpha=0.2)
    plt.tight_layout()
 
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figura guardada en: {save_path}")
 
    plt.show()


def hit_rate_at_k_trimestral_v2(
    df_preds: pd.DataFrame,
    k_values: list[int],
    verbose: bool = True,
) -> dict:
    """
    Versión revisada de hit_rate_at_k_trimestral con denominador corregido.
 
    DIFERENCIA RESPECTO A LA VERSIÓN ORIGINAL:
    En hit_rate_at_k_trimestral, el denominador de cada trimestre incluye
    todas las entidades con failed=1 ese trimestre, lo que contabiliza
    varias veces el mismo banco a lo largo de los trimestres en que su
    bandera está propagada. Esto sesga la métrica de dos formas: un banco
    correctamente detectado en t puede no estarlo en t+1 simplemente porque
    su señal de deterioro ya ha sido absorbida por la dinámica del sistema,
    penalizando al modelo por algo que no es un fallo discriminativo; y un
    banco no detectado en su primer trimestre de alerta puede aparecer
    como "acierto" en un trimestre posterior donde la señal ya es trivial,
    inflando artificialmente el Hit-Rate.
 
    CRITERIO CORREGIDO:
    Cada CERT contribuye al denominador exactamente una vez: en el trimestre
    de su PRIMERA activación de la bandera failed=1, que es el momento más
    próximo al onset real del proceso de deterioro y el que tiene mayor
    valor informativo para el regulador (detectar la entidad cuando el
    deterioro acaba de manifestarse, no cuando lleva trimestres en alerta).
    El numerador comprueba si esa entidad, en ese trimestre de primera
    activación, estaba dentro del top-K del ranking del modelo.
 
    CONSECUENCIA SOBRE LA INTERPRETACIÓN:
    La métrica resultante mide la capacidad del modelo para identificar el
    onset del deterioro de una entidad en el trimestre en que la señal
    regulatoria se activa por primera vez, evaluado contra el universo
    completo de entidades activas ese trimestre. Es una medida más exigente
    y más limpia que la versión original: cada banco tiene una sola
    oportunidad de ser detectado, en el momento en que la detección tiene
    mayor valor para el supervisor. Los trimestres posteriores de
    propagación de la bandera no contaminan el denominador.
 
    LIMITACIÓN RESIDUAL (no resoluble en el baseline):
    La primera activación de failed=1 tampoco garantiza que corresponda a
    una quiebra real en sentido jurídico estricto: puede ser el inicio de
    un proceso de fusión o absorción. La distinción definitiva requiere
    verificar la persistencia del nodo en el grafo relacional, análisis
    reservado para el modelo Híbrido.
 
    La estructura del dict devuelto es idéntica a hit_rate_at_k_trimestral
    para facilitar la comparación directa entre ambas versiones.
    """
    assert len(k_values) > 0, "k_values no puede estar vacío."
    assert all(isinstance(k, int) and k > 0 for k in k_values), \
        "k_values debe ser una lista de enteros positivos."
 
    # Primera activación de failed=1 por CERT
    first_failure = (
        df_preds[df_preds["failed"] == 1]
        .sort_values("period_end", key=lambda s: s.map(_parse_period))
        .groupby("CERT", sort=False)["period_end"]
        .first()
        .reset_index()
        .rename(columns={"period_end": "first_failure_period"})
    )
    # Indexado por trimestre: qué CERTs tienen su primera alerta ese trimestre
    first_by_period = first_failure.groupby("first_failure_period")["CERT"].apply(set).to_dict()
 
    results = {}
 
    for k in k_values:
        per_period_rows = []
        watchlist_rows = []
 
        for period, group in df_preds.groupby("period_end", sort=True):
            n_active = len(group)
            onset_certs = first_by_period.get(period, set())
            n_onset = len(onset_certs)
 
            ranked = group.sort_values("prob", ascending=False).reset_index(drop=True)
            top_k_certs = set(ranked.head(k)["CERT"].tolist())
 
            if n_onset == 0:
                watchlist_rows.append({
                    "period_end": period,
                    "n_active": n_active,
                    "top_k_certs": ranked.head(k)["CERT"].tolist(),
                    "top_k_probs": ranked.head(k)["prob"].round(4).tolist(),
                })
                continue
 
            n_hits = len(onset_certs & top_k_certs)
            per_period_rows.append({
                "period_end": period,
                "n_active": n_active,
                "n_onset": n_onset,
                "onset_certs": sorted(onset_certs),
                "n_hits": n_hits,
                "hit_rate": n_hits / n_onset,
            })
 
        df_per_period = pd.DataFrame(per_period_rows)
        df_watchlist = pd.DataFrame(watchlist_rows)
        macro_avg = df_per_period["hit_rate"].mean() if len(df_per_period) else float("nan")
 
        results[k] = {
            "macro_avg": macro_avg,
            "per_period": df_per_period,
            "watchlist_zero_onset_quarters": df_watchlist,
        }
 
        if verbose:
            print("\n" + "=" * 60)
            print(f"HIT-RATE@{k} TRIMESTRAL v2 (onset corregido)")
            print("=" * 60)
            print(f"Entidades únicas con onset        : {len(first_failure)}")
            print(f"Trimestres con onset registrado   : {len(df_per_period)}")
            print(f"Trimestres sin onset              : {len(df_watchlist)}  (excluidos del macro-promedio)")
            print(f"Hit-Rate@{k} v2 (macro-promedio)  : {macro_avg:.4f}")
            if len(df_per_period):
                cols = ["period_end", "n_active", "n_onset", "n_hits", "hit_rate"]
                print(df_per_period[cols].to_string(index=False))
            print("=" * 60)
 
    return results
 
 
 