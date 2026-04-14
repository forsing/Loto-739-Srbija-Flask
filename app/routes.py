import pandas as pd
from flask import Blueprint, render_template, request, jsonify, current_app
from pathlib import Path
from datetime import datetime

from .services.dataset import read_table, normalize_any, get_top_numbers, get_top_combinations
from .services.experiment import (
    run_experiment,
    summarize_experiment_results,
    predict_next_combo,
)

main_bp = Blueprint("main", __name__)

@main_bp.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        error=None,
        results=None,
        last_uploaded_file=None,
        form_state={
            "lottery": "loto739",
            "file_format": "loto739_num",
            "split_ratio": "70_30",
        },
        status="idle",
        best_summary=None,
        next_combo=None,
    )

@main_bp.route("/run", methods=["POST"])
def run():
    error = None
    results = None
    last_uploaded_file = None
    best_summary = None
    next_combo = None
    lottery = request.form.get("lottery", "loto739")
    file_format = request.form.get("file_format", "loto739_num")
    split_ratio = request.form.get("split_ratio", "70_30")

    form_state = {
        "lottery": lottery,
        "file_format": file_format,
        "split_ratio": split_ratio,
    }

    upload_dir = Path(main_bp.root_path).resolve().parent / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    outputs_dir = Path(main_bp.root_path).resolve().parent / "outputs"
    normalized_latest_path = outputs_dir / "normalized_latest.csv"

    file = request.files.get("dataset")
    df_norm = None

    try:
        if file and file.filename:
            safe_name = file.filename.replace(" ", "_")
            last_uploaded_file = safe_name
            saved_path = upload_dir / safe_name
            file.save(saved_path)

            df_raw = read_table(saved_path, file_format=file_format)
            df_norm = normalize_any(df_raw, lottery=lottery, file_format=file_format)

        elif normalized_latest_path.exists():
            df_norm = read_table(normalized_latest_path)
            if lottery == "loto739" and "n7" not in df_norm.columns:
                raise ValueError(
                    "Sačuvani podaci nisu Loto 7/39 (nema kolone n7). Otpremite CSV ponovo."
                )

        else:
            default_csv = current_app.config.get("DEFAULT_LOTTO739_CSV")
            p = Path(default_csv) if default_csv else None
            if p and p.exists():
                df_raw = read_table(p, file_format=file_format)
                df_norm = normalize_any(df_raw, lottery=lottery, file_format=file_format)
                last_uploaded_file = p.name
            else:
                error = (
                    "Otpremite CSV ili stavite GHQ/data/loto7hh_4596_k29.csv (Num1–Num7)."
                )
                return render_template(
                    "index.html",
                    error=error,
                    results=None,
                    last_uploaded_file=last_uploaded_file,
                    form_state=form_state,
                    status="idle",
                    best_summary=None,
                    next_combo=None,
                )

        results = run_experiment(
            df_norm,
            lottery,
            split_ratio=split_ratio
        )
        best_summary = summarize_experiment_results(results)

        best_name = None
        if best_summary and best_summary.get("best_overall"):
            best_name = best_summary["best_overall"].get("model")
        next_combo = predict_next_combo(df_norm, lottery, best_name)

        _save_outputs(df_norm, results, lottery)

        status = "done"

    except Exception as exc:
        error = str(exc)
        status = "idle"

    return render_template(
        "index.html",
        error=error,
        results=results,
        last_uploaded_file=last_uploaded_file,
        form_state=form_state,
        status=status,
        best_summary=best_summary if results else None,
        next_combo=next_combo,
    )

@main_bp.route("/top-numbers-api", methods=["POST"])
def top_numbers_api():
    outputs_dir = Path(main_bp.root_path).resolve().parent / "outputs"
    normalized_latest_path = outputs_dir / "normalized_latest.csv"

    if not normalized_latest_path.exists():
        return jsonify({
            "ok": False,
            "error": "Nema sačuvanih normalizovanih podataka. Prvo pokrenite eksperiment (Run)."
        })

    try:
        df_norm = read_table(normalized_latest_path)
        top_k = request.form.get("top_k", "5")

        try:
            top_k = int(top_k)
            if top_k < 1:
                top_k = 1
            elif top_k > 10:
                top_k = 10
        except ValueError:
            top_k = 5

        top_numbers = get_top_numbers(df_norm, k=top_k)

        if "n7" in df_norm.columns and df_norm["n7"].notna().any():
            analysis_label = "Loto 7/39"
        else:
            has_n6 = "n6" in df_norm.columns and df_norm["n6"].notna().any()
            analysis_label = "Viking Lotto" if has_n6 else "Eurojackpot"

        return jsonify({
            "ok": True,
            "analysis_label": analysis_label,
            "top_numbers": [
                {"number": number, "freq": freq}
                for number, freq in top_numbers
            ]
        })

    except Exception as exc:
        return jsonify({
            "ok": False,
            "error": str(exc)
        })

@main_bp.route("/top-combinations-api", methods=["POST"])
def top_combinations_api():
    outputs_dir = Path(main_bp.root_path).resolve().parent / "outputs"
    normalized_latest_path = outputs_dir / "normalized_latest.csv"

    if not normalized_latest_path.exists():
        return jsonify({
            "ok": False,
            "error": "Nema sačuvanih normalizovanih podataka. Prvo pokrenite eksperiment (Run)."
        })

    try:
        df_norm = read_table(normalized_latest_path)

        comb_size = request.form.get("comb_size", "2")
        top_k = request.form.get("top_k", "5")

        try:
            comb_size = int(comb_size)
            if comb_size < 2:
                comb_size = 2
            elif comb_size > 4:
                comb_size = 4
        except ValueError:
            comb_size = 2

        try:
            top_k = int(top_k)
            if top_k < 1:
                top_k = 1
            elif top_k > 10:
                top_k = 10
        except ValueError:
            top_k = 5

        top_combinations = get_top_combinations(df_norm, comb_size=comb_size, top_k=top_k)

        if "n7" in df_norm.columns and df_norm["n7"].notna().any():
            analysis_label = "Loto 7/39"
        else:
            has_n6 = "n6" in df_norm.columns and df_norm["n6"].notna().any()
            analysis_label = "Viking Lotto" if has_n6 else "Eurojackpot"

        return jsonify({
            "ok": True,
            "analysis_label": analysis_label,
            "top_combinations": [
                {"combo": list(combo), "freq": freq}
                for combo, freq in top_combinations
            ]
        })

    except Exception as exc:
        return jsonify({
            "ok": False,
            "error": str(exc)
        })

def _timestamp():
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S.") + f"{int(now.microsecond / 1000):03d}"

def _save_outputs(df_norm, results, lottery):
    outputs_dir = current_app.config["OUTPUTS_DIR"]

    df_norm.to_csv(outputs_dir / "normalized_latest.csv", index=False)

    df_res = pd.DataFrame(results)
    df_res.to_csv(outputs_dir / "results_latest.csv", index=False)

    history_path = outputs_dir / "results_history.csv"
    df_res_with_meta = df_res.copy()
    df_res_with_meta["timestamp"] = _timestamp()
    df_res_with_meta["lottery"] = lottery

    if history_path.exists():
        df_old = pd.read_csv(history_path)
        df_all = pd.concat([df_old, df_res_with_meta], ignore_index=True)
        df_all.to_csv(history_path, index=False)
    else:
        df_res_with_meta.to_csv(history_path, index=False)