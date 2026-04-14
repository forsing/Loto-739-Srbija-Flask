from flask import Flask
from pathlib import Path
from datetime import datetime

def create_app():
    # Flask aplikacija i osnovna konfiguracija
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "lottelligence-secret-key"

    # Jinja filter: format dates as dd.mm.yyyy
    @app.template_filter("ddmmyyyy")
    def ddmmyyyy(value):
        if value is None:
            return ""
        # datetime/date objects
        if hasattr(value, "strftime"):
            return value.strftime("%d.%m.%Y")
        # strings like "YYYY-MM-DD" or "YYYY-MM-DD ..."
        s = str(value)[:10]
        try:
            return datetime.strptime(s, "%Y-%m-%d").strftime("%d.%m.%Y")
        except ValueError:
            return ""

    # Osnovni direktorijumi (Lottelligence-main)
    base_dir = Path(__file__).resolve().parent.parent
    outputs_dir = base_dir / "outputs"
    uploads_dir = base_dir / "uploads"
    # Podrazumevani Loto 7/39: loto7hh (Num1–Num7) — usklađeno sa podrazumevanim formatom u UI
    ghq_root = base_dir.parent.parent
    app.config["DEFAULT_LOTTO739_CSV"] = ghq_root / "data" / "loto7hh_4596_k29.csv"

    # Kreiranje foldera
    outputs_dir.mkdir(parents=True, exist_ok=True)
    uploads_dir.mkdir(parents=True, exist_ok=True)

    app.config["BASE_DIR"] = base_dir
    app.config["OUTPUTS_DIR"] = outputs_dir
    app.config["UPLOAD_DIR"] = uploads_dir

    from .routes import main_bp
    app.register_blueprint(main_bp)

    return app