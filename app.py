from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from flask import (
    Flask,
    flash,
    g,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATA_DIR", str(BASE_DIR))).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_PATH = DATA_DIR / "app.db"
RAW_DATA_DIR = DATA_DIR / "data" / "raw"
ALLOWED_EXTENSIONS = {".xlsx", ".csv"}

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-key")


@dataclass
class DashboardMetrics:
    total_members: int
    avg_attendance: float
    avg_repurchase: float
    avg_csat: float
    performance_score: float
    judgment: str


def get_db() -> sqlite3.Connection:
    if "db" not in g:
        g.db = sqlite3.connect(DATABASE_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_: Any) -> None:
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db() -> None:
    db = sqlite3.connect(DATABASE_PATH)
    cur = db.cursor()
    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS bodycodi_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            member_name TEXT NOT NULL,
            branch TEXT,
            attendance_rate REAL NOT NULL,
            repurchase_rate REAL NOT NULL,
            csat_score REAL NOT NULL,
            month TEXT,
            uploaded_file TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS naver_place_accounts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            place_id TEXT,
            username TEXT NOT NULL,
            password TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        );

        CREATE TABLE IF NOT EXISTS naver_place_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            place_id TEXT,
            payload_json TEXT NOT NULL,
            fetched_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        );

        CREATE TABLE IF NOT EXISTS ingested_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            checksum TEXT NOT NULL,
            row_count INTEGER NOT NULL,
            ingested_at TEXT NOT NULL,
            UNIQUE(filename, checksum)
        );
        """
    )
    db.commit()
    db.close()


init_db()


def _allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def _file_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, str):
        value = value.strip().replace("%", "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_month(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.strftime("%Y-%m")
    return str(value).strip()


def _load_dataframe(file_path: Path) -> pd.DataFrame:
    if file_path.suffix.lower() == ".csv":
        return pd.read_csv(file_path)
    return pd.read_excel(file_path)


def _normalize_columns(df: pd.DataFrame) -> dict[str, str]:
    df.columns = [str(col).strip() for col in df.columns]
    return {col.lower(): col for col in df.columns}


def ingest_excel_files(file_paths: Iterable[Path] | None = None) -> dict[str, Any]:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    db = get_db()

    processed_files = []
    skipped_files = []
    invalid_files = []
    inserted_rows = 0
    skipped_rows = 0

    if file_paths is None:
        file_paths = sorted(
            list(RAW_DATA_DIR.glob("*.xlsx")) + list(RAW_DATA_DIR.glob("*.csv"))
        )
    else:
        file_paths = [Path(path) for path in file_paths]

    expected = {
        "member_name",
        "branch",
        "attendance_rate",
        "repurchase_rate",
        "csat_score",
        "month",
    }

    for file_path in file_paths:
        if not file_path.exists() or not _allowed_file(file_path.name):
            continue

        try:
            checksum = _file_checksum(file_path)
        except OSError:
            invalid_files.append(file_path.name)
            continue

        already_ingested = db.execute(
            "SELECT 1 FROM ingested_files WHERE filename = ? AND checksum = ?",
            (file_path.name, checksum),
        ).fetchone()
        if already_ingested:
            skipped_files.append(file_path.name)
            continue

        try:
            df = _load_dataframe(file_path)
        except Exception:
            invalid_files.append(file_path.name)
            continue

        if df.empty:
            invalid_files.append(file_path.name)
            continue

        col_map = _normalize_columns(df)
        if not expected.issubset(col_map):
            invalid_files.append(file_path.name)
            continue

        inserted_for_file = 0
        for _, row in df.iterrows():
            member_name = _safe_text(row[col_map["member_name"]])
            if not member_name:
                skipped_rows += 1
                continue

            attendance = _safe_float(row[col_map["attendance_rate"]])
            repurchase = _safe_float(row[col_map["repurchase_rate"]])
            csat = _safe_float(row[col_map["csat_score"]])

            if attendance is None or repurchase is None or csat is None:
                skipped_rows += 1
                continue

            branch = _safe_text(row[col_map["branch"]])
            month = _format_month(row[col_map["month"]])

            db.execute(
                """
                INSERT INTO bodycodi_records (
                    member_name, branch, attendance_rate, repurchase_rate,
                    csat_score, month, uploaded_file, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    member_name,
                    branch,
                    attendance,
                    repurchase,
                    csat,
                    month,
                    file_path.name,
                    datetime.utcnow().isoformat(),
                ),
            )
            inserted_rows += 1
            inserted_for_file += 1

        db.execute(
            """
            INSERT INTO ingested_files (filename, checksum, row_count, ingested_at)
            VALUES (?, ?, ?, ?)
            """,
            (file_path.name, checksum, inserted_for_file, datetime.utcnow().isoformat()),
        )
        processed_files.append(file_path.name)

    db.commit()
    return {
        "processed_files": processed_files,
        "skipped_files": skipped_files,
        "invalid_files": invalid_files,
        "inserted_rows": inserted_rows,
        "skipped_rows": skipped_rows,
    }


def compute_metrics() -> DashboardMetrics:
    db = get_db()
    row = db.execute(
        """
        SELECT
            COUNT(*) AS total_members,
            AVG(attendance_rate) AS avg_attendance,
            AVG(repurchase_rate) AS avg_repurchase,
            AVG(csat_score) AS avg_csat
        FROM bodycodi_records
        """
    ).fetchone()

    total_members = int(row["total_members"] or 0)
    avg_attendance = float(row["avg_attendance"] or 0)
    avg_repurchase = float(row["avg_repurchase"] or 0)
    avg_csat = float(row["avg_csat"] or 0)

    performance_score = round(
        avg_attendance * 0.4 + avg_repurchase * 0.35 + avg_csat * 10 * 0.25, 2
    )

    if performance_score >= 85:
        judgment = "우수"
    elif performance_score >= 70:
        judgment = "보통"
    else:
        judgment = "개선 필요"

    return DashboardMetrics(
        total_members=total_members,
        avg_attendance=round(avg_attendance, 2),
        avg_repurchase=round(avg_repurchase, 2),
        avg_csat=round(avg_csat, 2),
        performance_score=performance_score,
        judgment=judgment,
    )


def login_required() -> bool:
    return bool(session.get("user_id"))


def get_user() -> sqlite3.Row | None:
    if not login_required():
        return None
    return get_db().execute("SELECT * FROM users WHERE id = ?", (session["user_id"],)).fetchone()


def fetch_naver_place_data(place_id: str) -> dict[str, Any]:
    # 실제 연동 대신 데모 목적의 더미 데이터
    return {
        "place_id": place_id,
        "review_count": 124,
        "average_rating": 4.6,
        "recent_keywords": ["친절", "깨끗함", "재방문"],
        "recommended_autofill": {
            "attendance_rate": 82.0,
            "repurchase_rate": 78.0,
            "csat_score": 4.5,
        },
    }


@app.route("/")
def index() -> str:
    if login_required():
        return redirect(url_for("dashboard"))
    return render_template("index.html")


@app.route("/signup", methods=["GET", "POST"])
def signup() -> str:
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]
        db = get_db()

        if not username or not password:
            flash("아이디와 비밀번호를 입력해주세요.")
            return redirect(url_for("signup"))

        try:
            db.execute(
                "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                (username, generate_password_hash(password), datetime.utcnow().isoformat()),
            )
            db.commit()
            flash("회원가입 완료! 로그인해주세요.")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("이미 사용 중인 아이디입니다.")
            return redirect(url_for("signup"))

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login() -> str:
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]
        db = get_db()
        user = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()

        if user and check_password_hash(user["password_hash"], password):
            session.clear()
            session["user_id"] = user["id"]
            flash("로그인되었습니다.")
            return redirect(url_for("dashboard"))

        flash("아이디 또는 비밀번호가 올바르지 않습니다.")
        return redirect(url_for("login"))

    return render_template("login.html")


@app.route("/logout")
def logout() -> str:
    session.clear()
    flash("로그아웃되었습니다.")
    return redirect(url_for("index"))


@app.route("/dashboard", methods=["GET", "POST"])
def dashboard() -> str:
    if not login_required():
        return redirect(url_for("login"))

    ingest_result = None
    if request.method == "POST":
        action = request.form.get("action")
        if action == "ingest":
            ingest_result = ingest_excel_files()
        elif action == "upload":
            upload_file = request.files.get("data_file")
            if not upload_file or upload_file.filename == "":
                flash("업로드할 파일을 선택해주세요.")
            elif not _allowed_file(upload_file.filename):
                flash("지원하지 않는 파일 형식입니다. (.xlsx, .csv)")
            else:
                RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
                filename = secure_filename(upload_file.filename)
                saved_path = RAW_DATA_DIR / filename
                upload_file.save(saved_path)
                ingest_result = ingest_excel_files([saved_path])
                flash("파일 업로드 및 반영이 완료되었습니다.")

    metrics = compute_metrics()

    db = get_db()
    monthly_rows = db.execute(
        """
        SELECT month,
               ROUND(AVG(attendance_rate),2) AS attendance,
               ROUND(AVG(repurchase_rate),2) AS repurchase,
               ROUND(AVG(csat_score),2) AS csat
        FROM bodycodi_records
        GROUP BY month
        ORDER BY month
        """
    ).fetchall()

    chart_labels = [row["month"] for row in monthly_rows]
    attendance_data = [row["attendance"] for row in monthly_rows]
    repurchase_data = [row["repurchase"] for row in monthly_rows]
    recent_records = db.execute(
        """
        SELECT member_name, month, attendance_rate
        FROM bodycodi_records
        ORDER BY id DESC
        LIMIT 8
        """
    ).fetchall()

    return render_template(
        "dashboard.html",
        metrics=metrics,
        ingest_result=ingest_result,
        recent_records=recent_records,
        chart_labels=json.dumps(chart_labels, ensure_ascii=False),
        attendance_data=json.dumps(attendance_data),
        repurchase_data=json.dumps(repurchase_data),
    )


@app.route("/naver-place", methods=["GET", "POST"])
def naver_place() -> str:
    if not login_required():
        return redirect(url_for("login"))

    db = get_db()
    user = get_user()
    fetched_data = None

    if request.method == "POST":
        action = request.form.get("action")

        if action == "save_credential":
            place_id = request.form["place_id"]
            username = request.form["username"]
            password = request.form["password"]
            db.execute(
                """
                INSERT INTO naver_place_accounts (user_id, place_id, username, password, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (user["id"], place_id, username, password, datetime.utcnow().isoformat()),
            )
            db.commit()
            flash("네이버 플레이스 로그인 정보가 저장되었습니다.")

        if action == "fetch_data":
            place_id = request.form["place_id"]
            fetched_data = fetch_naver_place_data(place_id)
            db.execute(
                """
                INSERT INTO naver_place_cache (user_id, place_id, payload_json, fetched_at)
                VALUES (?, ?, ?, ?)
                """,
                (user["id"], place_id, json.dumps(fetched_data, ensure_ascii=False), datetime.utcnow().isoformat()),
            )
            db.commit()
            flash("네이버 플레이스 데이터를 조회했습니다. 자동입력 추천값을 확인하세요.")

        if action == "autofill_to_bodycodi":
            member_name = request.form["member_name"]
            branch = request.form["branch"]
            month = request.form["month"]
            attendance = float(request.form["attendance_rate"])
            repurchase = float(request.form["repurchase_rate"])
            csat = float(request.form["csat_score"])

            db.execute(
                """
                INSERT INTO bodycodi_records (
                    member_name, branch, attendance_rate, repurchase_rate, csat_score,
                    month, uploaded_file, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    member_name,
                    branch,
                    attendance,
                    repurchase,
                    csat,
                    month,
                    "naver_autofill",
                    datetime.utcnow().isoformat(),
                ),
            )
            db.commit()
            flash("네이버 플레이스 추천값이 바디코디 데이터로 자동입력되었습니다.")

    recent_cache = db.execute(
        """
        SELECT place_id, payload_json, fetched_at
        FROM naver_place_cache
        WHERE user_id = ?
        ORDER BY fetched_at DESC
        LIMIT 1
        """,
        (user["id"],),
    ).fetchone()

    if not fetched_data and recent_cache:
        fetched_data = json.loads(recent_cache["payload_json"])

    return render_template("naver_place.html", fetched_data=fetched_data)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
