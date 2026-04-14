import numpy as np

SEED = 39

# Sklearn komponentes: 
# SGDClassifier: loģistiskā regresija ar SGD optimizāciju 
# RandomForestClassifier: ansambļa modelis ar daudziem nejaušiem kokiem 
# GradientBoostingClassifier: vienkāršota boosting pieeja (fallback XGBoost vietā) 
# OneVsRestClassifier: ļauj trenēt vienu bināru modeli katram skaitlim (multi-label) 
# StandardScaler: stabilizē SGD svaru dinamiku 
# make_pipeline: apvieno scaler + modeli vienā ķēdē
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Mēģina importēt xgboost, ja nav – izmanto fallback
try:
    from xgboost import XGBClassifier  # type: ignore
    HAS_XGB = True
except Exception:
    XGBClassifier = None  # type: ignore
    HAS_XGB = False

def build_logreg_sgd():
    # Izveido stabilu loģistiskās regresijas modeli ar SGD
    # Stohastiskā gradienta metode - metode, kas atjaunina modeļa svarus, izmantojot nejauši izvēlētus datu punktus
    # StandardScaler stabilizē svaru dinamiku
    # Mazāks solis (eta0) novērš svaru eksplodēšanu
    # Regulārzācija (alpha) samazina overflow risku

    base = SGDClassifier(
        loss="log_loss",
        max_iter=300,
        tol=1e-3,
        learning_rate="constant",
        eta0=0.01,          # mazs solis -> stabilāka konverģence
        alpha=0.0005,       # neliela regulārzācija
        random_state=SEED,
    )

    # Scaler + SGD pipeline
    pipeline = make_pipeline(
        StandardScaler(with_mean=False),  # saglabā bināro sparsity
        base
    )

    model = OneVsRestClassifier(pipeline)
    return model

def build_random_forest():
    # Izveido RandomForest modeli
    # Darbojas labi ar nelineārām sakarībām
    # Nav nepieciešama skalēšana
    # Paralēlizējams (n_jobs=-1)

    base = RandomForestClassifier(
        n_estimators=100,
        random_state=SEED,
        n_jobs=1,
    )
    model = OneVsRestClassifier(base)
    return model

def build_xgboost_like():
    # Izveido XGBoost vai fallback GradientBoosting modeli

    if HAS_XGB:
        # Pilnais XGBoost (ja instalēts)
        base = XGBClassifier(            # type: ignore[operator]
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            n_jobs=1,
        )
        model = OneVsRestClassifier(base)
    else:
        # # Fallback: vienkāršots boosting
        base = GradientBoostingClassifier(random_state=SEED)
        model = OneVsRestClassifier(base)

    return model

def fit_and_predict(model, X_train, Y_train, X_test):
    # Apmāca modeli un atgriež paredzētās varbūtības (matrica ar izmēru [n_samples, max_num])
    # OneVsRestClassifier atgriež sarakstu ar (n_samples, 2) — tiek paņemta p(y=1)

    model.fit(X_train, Y_train)

    proba = model.predict_proba(X_test)

    # Ja OneVsRest atgriež sarakstu ar (n_samples, 2)
    if isinstance(proba, (list, tuple)):
        probs = [p[:, 1] for p in proba]  # p(y=1)
        proba_matrix = np.vstack(probs).T
    else:
        proba_matrix = proba

    return proba_matrix