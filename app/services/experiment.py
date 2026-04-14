import random
import numpy as np
import pandas as pd

# Sklearn metrikas modeļu novērtēšanai: 
# log_loss: mēra prognožu ticamību (jo mazāks, jo labāk) 
# mean_squared_error: Brier score (jo mazāks, jo labāk)
from sklearn.metrics import log_loss, mean_squared_error

# Izslēdz brīdinājumus par matricas reizināšanu
import warnings 
warnings.filterwarnings("ignore", message=".*matmul.*", category=RuntimeWarning)

# Modeļu būvēšana un prognozēšana
from .models import SEED, build_logreg_sgd, build_random_forest, build_xgboost_like, fit_and_predict

# Loto 7/39: koliko uzastopnih žreba u vektoru (ne samo poslednji) + freq/gap iz cele istorije do tog trenutka
_LOOKBACK = 5


def _set_seed():
    random.seed(SEED)
    np.random.seed(SEED)


def run_experiment(df_norm: pd.DataFrame, lottery: str, split_ratio: str = "70_30"):
    # Izpilda eksperimentu ar trim modeļiem (LogReg, RandomForest, XGBoost-like)
    # Izmanto lagged features: prev_vec -> curr_vec
    # Atgriež metrikas un informāciju par treniņu/testu periodiem

    # Nosaka loterijas parametrus
    if lottery == "loto739":
        max_num = 39
        k_main = 7
    elif lottery == "viking":
        max_num = 48
        k_main = 6
    elif lottery == "euro":
        max_num = 50
        k_main = 5
    else:
        raise ValueError("Nepoznat tip lutrije za eksperiment.")

    _set_seed()

    df_feat = _prepare_lagged_features(df_norm, max_num=max_num)
    df_feat = df_feat.sort_values("date").reset_index(drop=True)

    n = len(df_feat)
    if n < 10:
        raise ValueError("Nedovoljno redova posle lag obrade (potrebno bar 10).")

    use_rich = lottery == "loto739"
    if use_rich:
        V = _draw_matrix_sorted(df_norm, max_num)
        X_all = _rich_X_from_V(V, max_num, lookback=_LOOKBACK)
        if X_all.shape[0] != n:
            raise ValueError("Neusklađenost dimenzija obeležja.")
    else:
        X_all = np.stack(df_feat["prev_vec"].tolist())

    # % treniņam, % testam
    split_map = {
        "50_50": 0.50,
        "55_45": 0.55,
        "60_40": 0.60,
        "65_35": 0.65,
        "70_30": 0.70,
        "75_25": 0.75,
        "80_20": 0.80,
    }

    train_ratio = split_map.get(split_ratio, 0.70)

    split_idx = int(n * train_ratio)
    train = df_feat.iloc[:split_idx]
    test = df_feat.iloc[split_idx:]

    tr_slice = slice(0, split_idx)
    te_slice = slice(split_idx, n)
    X_train = X_all[tr_slice]
    X_test = X_all[te_slice]
    Y_train = np.stack(train["curr_vec"].tolist())
    Y_test = np.stack(test["curr_vec"].tolist())

    # Datumu diapazoni (informatīvi)
    train_date_from = train["date"].min().date().isoformat()
    train_date_to = train["date"].max().date().isoformat()
    test_date_from = test["date"].min().date().isoformat()
    test_date_to = test["date"].max().date().isoformat()

    results = []

    # Modeļu saraksts
    models = [
        ("logreg_sgd", build_logreg_sgd()),
        ("random_forest", build_random_forest()),
        ("xgboost", build_xgboost_like()),
    ]

    # Izpilda katru modeli
    for name, model in models:
        proba = fit_and_predict(model, X_train, Y_train, X_test)

        # Aizsardzība pret 0 un 1 (logloss nevar aprēķināt)
        proba_clipped = np.clip(proba, 1e-6, 1 - 1e-6)

        # Metrikas
        ll = log_loss(Y_test.ravel(), proba_clipped.ravel())
        brier = mean_squared_error(Y_test.ravel(), proba_clipped.ravel())
        hit_k_main = _hit_at_k(Y_test, proba_clipped, k=k_main)
        hit_10 = _hit_at_k(Y_test, proba_clipped, k=10)

        # Rezultātu rinda
        res = {
            "model": name,
            "logloss": float(ll),
            "brier": float(brier),
            "hit_k_main": float(hit_k_main),
            "hit_10": float(hit_10),
            "k_main": int(k_main),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_date_from": train_date_from,
            "train_date_to": train_date_to,
            "test_date_from": test_date_from,
            "test_date_to": test_date_to,
            "split_ratio": split_ratio,
        }
        results.append(res)

    return results


def predict_next_combo(
    df_norm: pd.DataFrame,
    lottery: str,
    best_model_name: str | None = None,
) -> list[int] | None:
    """Predlog: prozor od više žreba + freq/razmak iz cele istorije; prosek verovatnoća tri modela."""
    _ = best_model_name  # API ostaje; uvek se koristi ansambl tri modela
    if lottery != "loto739":
        return None
    _set_seed()
    max_num = 39
    k_main = 7
    df_feat = _prepare_lagged_features(df_norm, max_num=max_num)
    df_feat = df_feat.sort_values("date").reset_index(drop=True)
    if len(df_feat) < 10:
        return None

    V = _draw_matrix_sorted(df_norm, max_num)
    X_train = _rich_X_from_V(V, max_num, lookback=_LOOKBACK)
    Y_train = np.stack(df_feat["curr_vec"].tolist())
    if X_train.shape[0] != len(df_feat):
        return None

    X_pred = _rich_X_single_future(V, max_num, lookback=_LOOKBACK)

    p_sum = np.zeros(max_num, dtype=np.float64)
    for build in (build_logreg_sgd, build_random_forest, build_xgboost_like):
        model = build()
        proba = fit_and_predict(model, X_train, Y_train, X_pred)
        p_sum += np.asarray(proba).ravel()
    p = p_sum / 3.0

    hist = V.sum(axis=0)
    hist_n = hist / (hist.sum() + 1e-9)
    combined = p + 1e-9 * (hist_n / (hist_n.max() + 1e-9))
    top_idx = np.argsort(combined, kind="stable")[-k_main:]
    return sorted(int(i) + 1 for i in top_idx)


def summarize_experiment_results(results: list[dict]) -> dict | None:
    # Veido kopsavilkumu par labāko rezultātu pašreizējā eksperimenta ietvaros
    # Atgriež:
    # - labāko modeli katrai metrikai
    # - kopējo labāko modeli pēc rangu summas
    # - split informāciju
    # - labākā varianta metriku vērtības

    if not results:
        return None

    # Drošības filtrs: ņem tikai korektas rindas ar visām vajadzīgajām metrikām
    valid_results = []
    for row in results:
        if (
            "model" in row
            and "logloss" in row
            and "brier" in row
            and "hit_k_main" in row
            and "hit_10" in row
        ):
            valid_results.append(row)

    if not valid_results:
        return None

    # Labākie pēc atsevišķām metrikām
    best_logloss = min(valid_results, key=lambda r: r["logloss"])
    best_brier = min(valid_results, key=lambda r: r["brier"])
    best_hit_k_main = max(valid_results, key=lambda r: r["hit_k_main"])
    best_hit_10 = max(valid_results, key=lambda r: r["hit_10"])

    # Rangi katrai metrikai
    ranked_logloss = sorted(valid_results, key=lambda r: r["logloss"])
    ranked_brier = sorted(valid_results, key=lambda r: r["brier"])
    ranked_hit_k_main = sorted(valid_results, key=lambda r: r["hit_k_main"], reverse=True)
    ranked_hit_10 = sorted(valid_results, key=lambda r: r["hit_10"], reverse=True)

    # Summē rangu vietas katram modelim
    rank_sums: dict[str, int] = {}

    for idx, row in enumerate(ranked_logloss, start=1):
        rank_sums[row["model"]] = rank_sums.get(row["model"], 0) + idx

    for idx, row in enumerate(ranked_brier, start=1):
        rank_sums[row["model"]] = rank_sums.get(row["model"], 0) + idx

    for idx, row in enumerate(ranked_hit_k_main, start=1):
        rank_sums[row["model"]] = rank_sums.get(row["model"], 0) + idx

    for idx, row in enumerate(ranked_hit_10, start=1):
        rank_sums[row["model"]] = rank_sums.get(row["model"], 0) + idx

    # Atrod kopējo labāko modeli pēc mazākās rangu summas
    best_overall = min(valid_results, key=lambda r: rank_sums.get(r["model"], 10**9))
    best_model_name = best_overall["model"]

    metric_ranks = {
        "logloss": next(
            idx for idx, row in enumerate(ranked_logloss, start=1)
            if row["model"] == best_model_name
        ),
        "brier": next(
            idx for idx, row in enumerate(ranked_brier, start=1)
            if row["model"] == best_model_name
        ),
        "hit_k_main": next(
            idx for idx, row in enumerate(ranked_hit_k_main, start=1)
            if row["model"] == best_model_name
        ),
        "hit_10": next(
            idx for idx, row in enumerate(ranked_hit_10, start=1)
            if row["model"] == best_model_name
        ),
    }

    summary = {
        "best_by_metric": {
            "logloss": {
                "model": best_logloss["model"],
                "value": float(best_logloss["logloss"]),
            },
            "brier": {
                "model": best_brier["model"],
                "value": float(best_brier["brier"]),
            },
            "hit_k_main": {
                "model": best_hit_k_main["model"],
                "value": float(best_hit_k_main["hit_k_main"]),
            },
            "hit_10": {
                "model": best_hit_10["model"],
                "value": float(best_hit_10["hit_10"]),
            },
        },
        "best_overall": {
            "model": best_overall["model"],
            "rank_sum": int(rank_sums[best_model_name]),
            "metric_ranks": metric_ranks,
            "logloss": float(best_overall["logloss"]),
            "brier": float(best_overall["brier"]),
            "hit_k_main": float(best_overall["hit_k_main"]),
            "hit_10": float(best_overall["hit_10"]),
            "k_main": int(best_overall["k_main"]),
            "split_ratio": best_overall["split_ratio"],
            "train_rows": int(best_overall["train_rows"]),
            "test_rows": int(best_overall["test_rows"]),
            "train_date_from": best_overall["train_date_from"],
            "train_date_to": best_overall["train_date_to"],
            "test_date_from": best_overall["test_date_from"],
            "test_date_to": best_overall["test_date_to"],
        },
    }

    return summary

def _draw_matrix_sorted(df_norm: pd.DataFrame, max_num: int) -> np.ndarray:
    """Matrica žreba (hronološki): svaki red je one-hot izvlačenje."""
    rows = []
    for _, row in df_norm.sort_values("date").iterrows():
        mains = [row["n1"], row["n2"], row["n3"], row["n4"], row["n5"]]
        if not pd.isna(row.get("n6", pd.NA)):
            mains.append(row["n6"])
        if "n7" in df_norm.columns and not pd.isna(row.get("n7", pd.NA)):
            mains.append(row["n7"])
        mains_clean = [int(x) for x in mains if not pd.isna(x)]
        v = np.zeros(max_num, dtype=np.float64)
        for m in mains_clean:
            if 1 <= m <= max_num:
                v[m - 1] = 1.0
        rows.append(v)
    return np.asarray(rows, dtype=np.float64)


def _rich_X_from_V(V: np.ndarray, max_num: int, lookback: int) -> np.ndarray:
    """Za svaki korak i: predviđa se žreb V[i+1]; u obeležjima ulaze V[i], V[i-1], …, frekv. i razmaci nad V[0]..V[i]."""
    n_d = V.shape[0]
    n_feat = n_d - 1
    if n_feat <= 0:
        return np.zeros((0, lookback * max_num + max_num + max_num), dtype=np.float64)
    d = lookback * max_num + max_num + max_num
    X = np.zeros((n_feat, d), dtype=np.float64)
    for i in range(n_feat):
        c = 0
        for lag in range(lookback):
            j = i - lag
            if j >= 0:
                X[i, c : c + max_num] = V[j]
            c += max_num
        hist = V[: i + 1].sum(axis=0)
        X[i, c : c + max_num] = hist / (hist.sum() + 1e-9)
        c += max_num
        gaps = np.zeros(max_num, dtype=np.float64)
        for b in range(max_num):
            last = -1
            for t in range(i, -1, -1):
                if V[t, b] > 0.5:
                    last = t
                    break
            gaps[b] = float(i - last) if last >= 0 else float(i + 10)
        X[i, c : c + max_num] = gaps / (gaps.max() + 1e-9)
    return X


def _rich_X_single_future(V: np.ndarray, max_num: int, lookback: int) -> np.ndarray:
    """Jedan red obeležja posle poslednjeg poznatog žreba (cela istorija u freq/gap)."""
    n_d = V.shape[0]
    i = n_d - 1
    d = lookback * max_num + max_num + max_num
    x = np.zeros((1, d), dtype=np.float64)
    c = 0
    for lag in range(lookback):
        j = i - lag
        if j >= 0:
            x[0, c : c + max_num] = V[j]
        c += max_num
    hist = V.sum(axis=0)
    x[0, c : c + max_num] = hist / (hist.sum() + 1e-9)
    c += max_num
    gaps = np.zeros(max_num, dtype=np.float64)
    for b in range(max_num):
        last = -1
        for t in range(i, -1, -1):
            if V[t, b] > 0.5:
                last = t
                break
        gaps[b] = float(i - last) if last >= 0 else float(i + 10)
    x[0, c : c + max_num] = gaps / (gaps.max() + 1e-9)
    return x


def _prepare_lagged_features(df_norm: pd.DataFrame, max_num: int) -> pd.DataFrame:
    # Svako izvlačenje -> one-hot vektor dužine max_num; prev_vec / curr_vec

    rows = []
    for _, row in df_norm.iterrows():
        mains = [row["n1"], row["n2"], row["n3"], row["n4"], row["n5"]]
        if not pd.isna(row.get("n6", pd.NA)):
            mains.append(row["n6"])
        if "n7" in df_norm.columns and not pd.isna(row.get("n7", pd.NA)):
            mains.append(row["n7"])
        mains_clean = [int(x) for x in mains if not pd.isna(x)]

        vec = np.zeros(max_num, dtype=int)
        for m in mains_clean:
            if 1 <= m <= max_num:
                vec[m - 1] = 1

        rows.append({"date": pd.to_datetime(row["date"]), "vec": vec})

    df_vec = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    prev_vecs = [None] + df_vec["vec"].tolist()[:-1]
    df_vec["prev_vec"] = prev_vecs
    df_vec["curr_vec"] = df_vec["vec"]

    return df_vec.dropna(subset=["prev_vec"]).reset_index(drop=True)

def _hit_at_k(Y_true: np.ndarray, proba: np.ndarray, k: int) -> float:
    # Aprēķina hit@k — cik bieži patiesie skaitļi ir starp top-k prognozētajiem

    hits = []
    for y, p in zip(Y_true, proba):
        top_idx = np.argsort(p, kind="stable")[-k:]
        top_set = set(top_idx.tolist())
        true_idx = set(np.where(y == 1)[0].tolist())
        inter = len(true_idx.intersection(top_set))
        hits.append(inter / float(k))
    return float(np.mean(hits))