# src/data/graph_builder.py
"""
Construcción del grafo dinámico bancario.

Para cada trimestre t produce un objeto torch_geometric.data.Data con:
    data.x          (|Vt|, 204)  atributos de nodo: 192 emb TabPFN + 12 estructurales
    data.edge_index (2, |Et|)    aristas en formato COO (holding company RSSDHCR)
    data.edge_attr  (|Et|, 1)    pesos binarios (1.0)
    data.y          (|Vt|,)      etiquetas failed
    data.cert       list[str]    CERTs en el mismo orden que data.x
    data.period     str          trimestre

Teoría:
    Las aristas modelan conglomerados financieros regulatorios.
    (i,j) in Et <=> RSSDHCR_i^t == RSSDHCR_j^t != NaN
    Esto captura exposición compartida al riesgo del high holder
    (Brunnermeier & Oehmke, 2013).

    Los atributos de nodo fusionan representación tabular de riesgo
    (TabPFN, 192 dims) con información estructural institucional (12 dims).
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from typing import Optional


# =============================================================================
# Columnas estructurales que entran como atributos de nodo
# =============================================================================

CAT_COLS = ['BKCLASS', 'INSTTYPE', 'CHRTAGNT', 'REGAGNT', 'STALP']
NUM_COLS = ['SIMS_LAT', 'SIMS_LONG', 'OFFDOM', 'OFFTOT', 'OFFSTATE']
BIN_COLS = ['DENOVO', 'METRO']

STRUCTURAL_COLS = CAT_COLS + NUM_COLS + BIN_COLS  # 12 columnas → s_i^t ∈ R^12
EMB_DIM  = 192
NODE_DIM = EMB_DIM + len(STRUCTURAL_COLS)  # 204


# =============================================================================
# Clase principal
# =============================================================================

class GraphBuilder:

    def __init__(
        self,
        panel_nodos: pd.DataFrame,
        embeddings: pd.DataFrame,
        labeled_panel: pd.DataFrame,
        device: str = 'cpu'
    ):
        self.nodos  = panel_nodos.copy()
        self.emb    = embeddings.copy()
        self.labels = labeled_panel.copy()
        self.device = device
        self.encoders: dict[str, LabelEncoder] = {}
        self._fitted = False

        # Tipos correctos
        self.nodos['CERT']  = self.nodos['CERT'].astype(str)
        self.emb['CERT']    = self.emb['CERT'].astype(str)
        self.labels['CERT'] = self.labels['CERT'].astype(str)

        # OFFDOM, OFFTOT son object (strings de enteros) — convertir
        for col in ['OFFDOM', 'OFFTOT','SIMS_LAT', 'SIMS_LONG']:
            self.nodos[col] = pd.to_numeric(self.nodos[col], errors='coerce')

        # RSSDHCR a entero nullable para comparación exacta
        self.nodos['RSSDHCR'] = self.nodos['RSSDHCR'].astype('Int64')

        self.periodos = sorted(self.emb['period'].unique())

    # -------------------------------------------------------------------------
    # Fit
    # -------------------------------------------------------------------------

    def fit(self):
        """
        Fit LabelEncoders sobre el panel completo antes de construir snapshots.
        Necesario para que la codificación sea consistente entre trimestres:
        el mismo valor siempre produce el mismo entero independientemente
        de qué categorías aparezcan en cada snapshot individual.

        La imputación también se calcula sobre el panel completo para evitar
        que estadísticos de un trimestre individual contaminen otros.
        """
        # LabelEncoder por columna categórica
        for col in CAT_COLS:
            le = LabelEncoder()
            valores = self.nodos[col].fillna('UNKNOWN').astype(str)
            le.fit(valores)
            self.encoders[col] = le

        self._impute_stats = {}

        # SIMS_LAT y SIMS_LONG — media por estado como diccionario + media global
        for coord in ['SIMS_LAT', 'SIMS_LONG']:
            self._impute_stats[coord + '_global'] = self.nodos[coord].mean()
            self._impute_stats[coord + '_stalp']  = (
                self.nodos.groupby('STALP')[coord].mean().to_dict()
            )

        # OFFDOM, OFFTOT, OFFSTATE — mediana global
        for col in ['OFFDOM', 'OFFTOT', 'OFFSTATE']:
            self._impute_stats[col] = self.nodos[col].median()

        # STALP — moda por CERT como diccionario {CERT: moda}
        self._stalp_moda_map = (
            self.nodos.groupby('CERT')['STALP']
            .agg(lambda x: x.mode()[0] if x.notna().any() else 'UNKNOWN')
            .to_dict()
        )

        self._fitted = True
        print(f'GraphBuilder fit completado.')
        print(f'  Categóricas codificadas : {CAT_COLS}')
        print(f'  Periodos disponibles    : {self.periodos[0]} → {self.periodos[-1]} ({len(self.periodos)} trim.)')

    # -------------------------------------------------------------------------
    # Construcción de un snapshot individual
    # -------------------------------------------------------------------------

    def build_snapshot(self, periodo: str) -> Optional[Data]:
        """
        Construye el snapshot Gt = (Vt, Et, Xt, At) para el trimestre t.

        Returns None si no hay embeddings para ese periodo.

        El orden de nodos en data.x es el mismo que en data.cert,
        garantizando que la fila i de data.x corresponde al banco
        data.cert[i] en el trimestre periodo.
        """
        assert self._fitted, "Llamar a fit() antes de build_snapshot()"

        # ── 1. Vt: bancos con embedding en este trimestre ──────────────────
        emb_t = self.emb[self.emb['period'] == periodo].copy()
        if len(emb_t) == 0:
            return None

        emb_cols = [c for c in emb_t.columns if c.startswith('emb_')]
        assert len(emb_cols) == EMB_DIM, \
            f"Esperadas {EMB_DIM} dims, encontradas {len(emb_cols)}"

        # ── 2. Join con panel_nodos para atributos estructurales ───────────
        nodos_t = self.nodos[self.nodos['period'] == periodo].copy()

        # SIMS_LAT y SIMS_LONG — media por estado, fallback a media global
        for coord in ['SIMS_LAT', 'SIMS_LONG']:
            media_global = self._impute_stats[coord + '_global']
            media_stalp  = self._impute_stats[coord + '_stalp']
            mask_na = nodos_t[coord].isna()
            if mask_na.any():
                fallback = nodos_t.loc[mask_na, 'STALP'].map(media_stalp).fillna(media_global)
                nodos_t.loc[mask_na, coord] = fallback

        # OFFDOM, OFFTOT, OFFSTATE — mediana global
        for col in ['OFFDOM', 'OFFTOT', 'OFFSTATE']:
            nodos_t[col] = nodos_t[col].fillna(self._impute_stats[col])

        # STALP — moda por CERT, fallback a UNKNOWN
        nodos_t['STALP'] = nodos_t.apply(
            lambda row: (
                self._stalp_moda_map.get(row['CERT'], 'UNKNOWN')
                if pd.isna(row['STALP']) else row['STALP']
            ),
            axis=1
        )

        # METRO: NaN = banco rural = 0
        nodos_t['METRO'] = nodos_t['METRO'].fillna(0.0)

        # Codificación categórica
        for col in CAT_COLS:
            vals = nodos_t[col].fillna('UNKNOWN').astype(str)
            nodos_t[col] = self.encoders[col].transform(vals)

        # Guardar failed antes del merge
        failed_vals = emb_t.set_index('CERT')['failed']

        # Join emb_t con nodos_t por CERT
        merged = emb_t.merge(
            nodos_t[['CERT'] + STRUCTURAL_COLS + ['RSSDHCR']],
            on='CERT',
            how='left'
        )

        # Reincorporar failed
        merged['failed'] = merged['CERT'].map(failed_vals)

        # Orden determinista por CERT para reproducibilidad
        merged = merged.sort_values('CERT').reset_index(drop=True)
        n_nodos = len(merged)

        # ── 3. Xt: matriz de atributos de nodo (|Vt|, 204) ────────────────
        emb_matrix    = merged[emb_cols].values.astype(np.float32)       # (n, 192)
        struct_matrix = merged[STRUCTURAL_COLS].values.astype(np.float32) # (n, 12)
        X = np.concatenate([emb_matrix, struct_matrix], axis=1)           # (n, 204)

        # ── 4. Et: aristas por RSSDHCR ────────────────────────────────────
        merged['node_idx'] = merged.index

        holding_mask   = merged['RSSDHCR'].notna()
        bancos_holding = merged[holding_mask][['node_idx', 'RSSDHCR']]

        src_list, dst_list = [], []

        for _, grupo in bancos_holding.groupby('RSSDHCR'):
            idxs = grupo['node_idx'].values
            if len(idxs) < 2:
                continue
            ii, jj  = np.meshgrid(idxs, idxs)
            mask     = ii != jj
            src_list.append(ii[mask])
            dst_list.append(jj[mask])

        if src_list:
            src = np.concatenate(src_list)
            dst = np.concatenate(dst_list)
        else:
            src = np.array([], dtype=np.int64)
            dst = np.array([], dtype=np.int64)

        edge_index = torch.tensor(
            np.stack([src, dst], axis=0),
            dtype=torch.long
        )
        edge_attr = torch.ones(edge_index.shape[1], 1, dtype=torch.float32)

        # ── 5. Etiquetas y ─────────────────────────────────────────────────
        y = torch.tensor(merged['failed'].values, dtype=torch.float32)

        # ── 6. Ensamblar objeto Data ───────────────────────────────────────
        data = Data(
            x          = torch.tensor(X, dtype=torch.float32),
            edge_index = edge_index,
            edge_attr  = edge_attr,
            y          = y,
            num_nodes  = n_nodos,
        )
        data.cert   = merged['CERT'].tolist()
        data.period = periodo

        return data

    # -------------------------------------------------------------------------
    # Construcción de todos los snapshots
    # -------------------------------------------------------------------------

    def build_all(self, verbose: bool = True) -> list[Data]:
        """
        Construye la secuencia completa de snapshots para todos los periodos
        disponibles en los embeddings.
        """
        assert self._fitted, "Llamar a fit() antes de build_all()"

        snapshots = []
        for periodo in self.periodos:
            data = self.build_snapshot(periodo)
            if data is None:
                continue
            snapshots.append(data)
            if verbose:
                n_aristas = data.edge_index.shape[1] // 2
                n_pos     = int(data.y.sum().item())
                print(
                    f'[{periodo}] nodos: {data.num_nodes:5d} | '
                    f'aristas: {n_aristas:6d} | '
                    f'positivos: {n_pos:3d} | '
                    f'dim_x: {data.x.shape[1]}'
                )

        print(f'\nTotal snapshots construidos: {len(snapshots)}')
        return snapshots

    # -------------------------------------------------------------------------
    # Serialización
    # -------------------------------------------------------------------------

    def save_snapshots(self, snapshots: list[Data], output_dir: Path):
        """
        Guarda cada snapshot como archivo .pt individual más un índice de periodos.
        Permite reanudar sin reconstruir si el proceso se interrumpe.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for data in snapshots:
            path = output_dir / f'snapshot_{data.period}.pt'
            torch.save(data, path)

        pd.Series([d.period for d in snapshots]).to_csv(
            output_dir / 'periodos_index.csv', index=False, header=False
        )
        print(f'Snapshots guardados en {output_dir}')

    @staticmethod
    def load_snapshots(output_dir: Path) -> list[Data]:
        """Carga snapshots guardados previamente."""
        output_dir = Path(output_dir)
        periodos   = pd.read_csv(
            output_dir / 'periodos_index.csv', header=None
        )[0].tolist()
        snapshots  = [
            torch.load(output_dir / f'snapshot_{p}.pt', weights_only=False)
            for p in periodos
        ]
        print(f'Cargados {len(snapshots)} snapshots desde {output_dir}')
        return snapshots