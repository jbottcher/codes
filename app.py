from pathlib import Path
from typing import Dict, List, Optional, Tuple
from fastapi import FastAPI
from pydantic import BaseModel, Field, validator
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

# =========================
# Configuração (opcional)
# =========================
# Se quiser inferir bounds e setpoints do Excel, informe:
EXCEL_PATH: Optional[Path] = None
EXCEL_SHEET: str = "digestor_sintetico"

# Exemplo (descomente e ajuste se quiser usar o Excel):
EXCEL_PATH = Path(r"C:\Users\jairb\OneDrive - Microsoft\Microsoft\Work\ATS\Accounts\JBS\digestor_sintetico_batelada_final.xlsx")

# =========================
# Modelo de entrada da API
# =========================
class OptimizeRequest(BaseModel):
    # Hiperparâmetros do SPEA2
    n_pop: int = Field(60, ge=10, description="Tamanho da população")
    n_archive: int = Field(30, ge=10, description="Tamanho do arquivo externo (elite)")
    generations: int = Field(25, ge=5, description="Número de gerações")
    pc: float = Field(0.9, ge=0.0, le=1.0, description="Probabilidade de cruzamento")
    pm: float = Field(0.2, ge=0.0, le=1.0, description="Probabilidade de mutação")

    # Setpoints/constantes do problema
    kappa_sp: float = Field(18.0, description="Setpoint/target de Kappa para o objetivo f1")
    rel_licor_madeira: float = Field(5.5, description="Relação licor/madeira média (proxy para rendimento)")

    # Limites das variáveis de decisão (se não enviar, uso defaults plausíveis)
    bounds: Optional[Dict[str, Tuple[float, float]]] = Field(
        default=None,
        description="Limites para variáveis de decisão: {var: [min, max]}"
    )

    @validator("bounds", pre=True, always=True)
    def default_bounds(cls, v):
        if v is None:
            # Defaults plausíveis (ajuste conforme sua realidade/dataset)
            return {
                "Temp_Bulk_C": [149.0, 160.0],
                "Carga_Alcalina_madeira": [14.0, 22.0],
                "Sulfidez": [22.0, 38.0],
                "Tempo_Coccao_min": [60.0, 180.0],
            }
        return v

# =========================
# Modelo de saída da API
# =========================
class Solution(BaseModel):
    Temp_Bulk_C: float
    Carga_Alcalina_madeira: float
    Sulfidez: float
    Tempo_Coccao_min: float
    Kappa: float
    Rendimento: float
    Erro_Kappa2: float
    Score_Proxy: float

class OptimizeResponse(BaseModel):
    pareto_size: int
    top15: List[Solution]

# =========================
# Simulação (EDO) e Métricas
# =========================
def simular_digestor(temp_bulk_c: float, carga_alcalina: float,
                     sulfidez: float, tempo_coccao_min: float) -> Tuple[float, float]:
    """
    EDO simplificada: dL/dt = -k * L, com k dependente de Temp, Carga e Sulfidez.
    Retorna (kappa_sim, L_end).
    """
    T_ref = 155.0
    a, b, c = 0.04, 0.03, 0.20
    k0 = 0.015

    k = k0 * np.exp(a * (temp_bulk_c - T_ref)) \
        * (1.0 + c * (carga_alcalina - 18)/18.0) \
        * (1.0 + b * (sulfidez - 30)/30.0)

    def dLdt(t, L):
        return -k * L

    L0 = [100.0]
    t_span = (0.0, float(tempo_coccao_min))
    t_eval = np.linspace(t_span[0], t_span[1], 200)

    sol = solve_ivp(dLdt, t_span, L0, t_eval=t_eval, method="RK45")
    L_end = float(sol.y[0, -1])

    theta = 3.3
    kappa_sim = float(np.clip(L_end / theta, 10, 30))
    return kappa_sim, L_end

def rendimento_estimado(kappa: float, relacao_licor_madeira: float = 5.5) -> float:
    return float(np.clip(48 + 0.30*(kappa - 18) - 0.05*(relacao_licor_madeira - 5.5), 40, 55))

# =========================
# Utilidades (SPEA2)
# =========================
def avaliar_objetivos(x: np.ndarray, kappa_sp: float, rel_licor_med: float):
    Tbulk, carga, sulf, tcoc = x
    kappa_sim, L_end = simular_digestor(Tbulk, carga, sulf, tcoc)
    rend_sim = rendimento_estimado(kappa_sim, relacao_licor_madeira=rel_licor_med)
    f1 = (kappa_sim - kappa_sp)**2
    f2 = -rend_sim
    extras = {"Kappa": kappa_sim, "Rendimento": rend_sim, "L_end": L_end}
    return np.array([f1, f2], dtype=float), extras

def domina(fa: np.ndarray, fb: np.ndarray) -> bool:
    return np.all(fa <= fb) and np.any(fa < fb)

def calcular_fitness_SPEA2(pop_objs: List[np.ndarray]):
    F = np.array(pop_objs)  # N x M
    N = F.shape[0]
    S = np.zeros(N, dtype=float)
    for i in range(N):
        for j in range(N):
            if i != j and domina(F[i], F[j]):
                S[i] += 1.0
    R = np.zeros(N, dtype=float)
    for i in range(N):
        dominadores = [j for j in range(N) if j != i and domina(F[j], F[i])]
        R[i] = np.sum(S[dominadores]) if dominadores else 0.0
    k_idx = int(np.sqrt(N)) or 1
    D = np.zeros((N, N), dtype=float)
    for i in range(N):
        di = np.linalg.norm(F - F[i], axis=1)
        D[i] = di
        D[i, i] = np.inf
    d_k = np.partition(D, k_idx, axis=1)[:, k_idx]
    density = 1.0 / (d_k + 2.0)
    fitness = R + density
    return fitness, density

def truncar_por_aglomeracao(indices: List[int], F: np.ndarray, Nmax: int) -> List[int]:
    idx = indices.copy()
    while len(idx) > Nmax:
        M = len(idx)
        subF = F[idx]
        D = np.full((M, M), np.inf, dtype=float)
        for i in range(M):
            for j in range(i+1, M):
                d = np.linalg.norm(subF[i] - subF[j])
                D[i, j] = D[j, i] = d
        i_min, j_min = np.unravel_index(np.argmin(D), D.shape)
        # remove o que estiver mais aglomerado
        mean_i = np.mean(D[i_min][np.isfinite(D[i_min])])
        mean_j = np.mean(D[j_min][np.isfinite(D[j_min])])
        remove = i_min if mean_i < mean_j else j_min
        del idx[remove]
    return idx

rng = np.random.default_rng(123)

def reparar(x: np.ndarray, bounds: Dict[str, Tuple[float, float]], var_order: List[str]):
    y = x.copy()
    for i, v in enumerate(var_order):
        lo, hi = bounds[v]
        y[i] = float(np.clip(y[i], lo, hi))
    return y

def cruzamento_aritmetico(p1: np.ndarray, p2: np.ndarray, pc: float,
                          bounds: Dict[str, Tuple[float, float]], var_order: List[str]):
    if rng.random() > pc:
        return p1.copy(), p2.copy()
    alpha = rng.uniform(0, 1, size=len(p1))
    c1 = alpha*p1 + (1-alpha)*p2
    c2 = alpha*p2 + (1-alpha)*p1
    return reparar(c1, bounds, var_order), reparar(c2, bounds, var_order)

def mutacao_gauss(x: np.ndarray, pm: float, sigma_frac: float,
                  bounds: Dict[str, Tuple[float, float]], var_order: List[str]):
    y = x.copy()
    for i, v in enumerate(var_order):
        if rng.random() < pm:
            lo, hi = bounds[v]
            sigma = (hi - lo)*sigma_frac
            y[i] = y[i] + rng.normal(0, sigma)
    return reparar(y, bounds, var_order)

def amostrar_inicial(N: int, bounds: Dict[str, Tuple[float, float]], var_order: List[str]):
    X = []
    for _ in range(N):
        x = [rng.uniform(*bounds[v]) for v in var_order]
        X.append(np.array(x, dtype=float))
    return X

# =========================
# FastAPI app
# =========================
app = FastAPI(title="Digestor Optimizer API (SPEA2 + EDO)")

@app.post("/optimize", response_model=OptimizeResponse)
def optimize(req: OptimizeRequest):
    # 1) Se quiser puxar defaults do Excel (opcional)
    if EXCEL_PATH and EXCEL_PATH.exists():
        try:
            df = pd.read_excel(EXCEL_PATH, sheet_name=EXCEL_SHEET, engine="openpyxl")
            numeric_df = df.select_dtypes(include="number")
            # Se bounds vieram vazios, inferir a partir dos dados
            if req.bounds is None or len(req.bounds) == 0:
                req.bounds = {
                    "Temp_Bulk_C": [float(numeric_df["Temp_Bulk_C"].min()), float(numeric_df["Temp_Bulk_C"].max())],
                    "Carga_Alcalina_madeira": [float(numeric_df["Carga_Alcalina_madeira"].min()), float(numeric_df["Carga_Alcalina_madeira"].max())],
                    "Sulfidez": [float(numeric_df["Sulfidez"].min()), float(numeric_df["Sulfidez"].max())],
                    "Tempo_Coccao_min": [float(numeric_df["Tempo_Coccao_min"].min()), float(numeric_df["Tempo_Coccao_min"].max())],
                }
            # kappa_sp/rel_licor_madeira do dataset, se desejar
            if "Kappa_Number" in numeric_df.columns:
                req.kappa_sp = float(numeric_df["Kappa_Number"].median())
            if "Relacao_Licor_Madeira_m3_BDt" in numeric_df.columns:
                req.rel_licor_madeira = float(numeric_df["Relacao_Licor_Madeira_m3_BDt"].median())
        except Exception:
            # Se falhar, segue com os valores enviados/defaults
            pass

    # 2) Ordem das variáveis de decisão
    var_order = ["Temp_Bulk_C", "Carga_Alcalina_madeira", "Sulfidez", "Tempo_Coccao_min"]
    for v in var_order:
        if v not in req.bounds:
            raise ValueError(f"Bounds ausente para variável '{v}'")

    # 3) Inicializar população
    P = amostrar_inicial(req.n_pop, req.bounds, var_order)
    objs, extras = [], []
    for x in P:
        f, m = avaliar_objetivos(x, req.kappa_sp, req.rel_licor_madeira)
        objs.append(f); extras.append(m)

    # 4) SPEA2 loop
    A: List[np.ndarray] = []
    A_objs: List[np.ndarray] = []
    A_ext: List[dict] = []

    for _gen in range(req.generations):
        P_all = (A + P) if len(A) > 0 else P
        Objs_all = (A_objs + objs) if len(A_objs) > 0 else objs
        Ext_all  = (A_ext + extras) if len(A_ext) > 0 else extras

        fitness, density = calcular_fitness_SPEA2(Objs_all)
        idx_all = list(range(len(P_all)))
        ord_all = sorted(idx_all, key=lambda i: (fitness[i], density[i]))
        A_idx = [i for i in ord_all if fitness[i] < np.median(fitness)]
        if len(A_idx) < req.n_archive:
            for i in ord_all:
                if i not in A_idx:
                    A_idx.append(i)
                if len(A_idx) >= req.n_archive:
                    break
        if len(A_idx) > req.n_archive:
            A_idx = truncar_por_aglomeracao(A_idx, np.array(Objs_all), req.n_archive)

        A = [P_all[i].copy() for i in A_idx]
        A_objs = [Objs_all[i].copy() for i in A_idx]
        A_ext  = [Ext_all[i].copy() for i in A_idx]

        # Nova geração
        fit_A, dens_A = calcular_fitness_SPEA2(A_objs)
        # Torneio
        pais_idx = []
        for _ in range(req.n_pop):
            cand = rng.choice(range(len(A)), size=2, replace=True)
            best = cand[np.argmin(fit_A[cand])]
            pais_idx.append(best)

        filhos = []
        for i in range(0, req.n_pop, 2):
            p1 = A[pais_idx[i % len(pais_idx)]]
            p2 = A[pais_idx[(i+1) % len(pais_idx)]]
            c1, c2 = cruzamento_aritmetico(p1, p2, req.pc, req.bounds, var_order)
            c1 = mutacao_gauss(c1, req.pm, 0.05, req.bounds, var_order)
            c2 = mutacao_gauss(c2, req.pm, 0.05, req.bounds, var_order)
            filhos.extend([c1, c2])
        P = filhos[:req.n_pop]

        objs, extras = [], []
        for x in P:
            f, m = avaliar_objetivos(x, req.kappa_sp, req.rel_licor_madeira)
            objs.append(f); extras.append(m)

    # 5) Montar TOP 15 (menor erro de Kappa, maior rendimento)
    rows = []
    for x, f, m in zip(A, A_objs, A_ext):
        rows.append({
            "Temp_Bulk_C": float(x[0]),
            "Carga_Alcalina_madeira": float(x[1]),
            "Sulfidez": float(x[2]),
            "Tempo_Coccao_min": float(x[3]),
            "Kappa": float(m["Kappa"]),
            "Rendimento": float(m["Rendimento"]),
            "Erro_Kappa2": float(f[0]),
            "Score_Proxy": float(-(f[0]) + (-f[1]))
        })

    df_sol = pd.DataFrame(rows)
    df_top15 = df_sol.sort_values(
        by=["Erro_Kappa2", "Rendimento"], ascending=[True, False]
    ).head(15)

    top15 = [
        Solution(**rec) for rec in df_top15.to_dict(orient="records")
    ]

    return OptimizeResponse(
        pareto_size=len(df_sol),
        top15=top15
    )

# (Opcional) endpoint de saúde
@app.get("/health")
def health():
    return {"status": "ok"}
