from __future__ import annotations

import hashlib
import json
import os
import re
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
BODYCODI_REQUIRED_COLUMNS = {
    "member_name",
    "attendance_rate",
    "repurchase_rate",
    "csat_score",
    "month",
}
BODYCODI_COLUMN_ALIASES = {
    "member_name": {
        "member_name",
        "membername",
        "회원명",
        "회원이름",
        "고객명",
        "이름",
        "회원",
    },
    "branch": {
        "branch",
        "지점",
        "센터",
        "클럽",
        "지부",
    },
    "attendance_rate": {
        "attendance_rate",
        "attendance",
        "attendance%",
        "출석률",
        "출석",
        "출석율",
    },
    "repurchase_rate": {
        "repurchase_rate",
        "repurchase",
        "repurchase%",
        "재구매율",
        "재구매",
        "재구매율(%)",
    },
    "csat_score": {
        "csat_score",
        "csat",
        "고객만족",
        "고객만족도",
        "만족도",
    },
    "month": {
        "month",
        "월",
        "년월",
        "기간",
        "정산월",
    },
}
SALES_REQUIRED_COLUMNS = {
    "payment_datetime",
    "payment_amount",
}
SALES_COLUMN_ALIASES = {
    "branch_name": {"지점명", "지점", "센터", "클럽", "branch"},
    "payment_datetime": {"결제일시", "결제일", "결제일자", "payment_datetime"},
    "member_name": {"회원명", "회원이름", "고객명", "이름", "member_name"},
    "contact": {"연락처", "전화번호", "휴대폰", "contact"},
    "sale_id": {"판매번호", "판매ID", "sale_id"},
    "product": {"판매상품", "상품명", "product"},
    "sale_status": {"판매상태", "상태", "sale_status"},
    "sale_price": {"판매가", "판매금액", "정가", "sale_price"},
    "payment_category": {"결제분류", "결제구분", "payment_category"},
    "payment_method": {"결제수단", "payment_method"},
    "payment_type": {"결제방법", "payment_type"},
    "deduction": {"공제", "공제금액", "deduction"},
    "payment_amount": {"결제금액", "결제액", "payment_amount"},
    "outstanding_amount": {"미수결제", "미수", "outstanding_amount"},
    "contract": {"전자계약서", "계약서", "contract"},
    "payment_request": {"결제요청", "payment_request"},
    "refund_method": {"환불수단", "refund_method"},
    "refund_amount": {"환불지급액", "환불금액", "refund_amount"},
    "sales_rep": {"판매담당자", "담당자", "sales_rep"},
    "memo": {"메모", "비고", "memo"},
    "points": {"포인트적립", "포인트 적립", "points"},
}


def _normalize_key(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[^0-9a-z가-힣]+", "", text)
    return text


NORMALIZED_BODYCODI_ALIASES = {
    canonical: {_normalize_key(alias) for alias in aliases}
    for canonical, aliases in BODYCODI_COLUMN_ALIASES.items()
}
NORMALIZED_SALES_ALIASES = {
    canonical: {_normalize_key(alias) for alias in aliases}
    for canonical, aliases in SALES_COLUMN_ALIASES.items()
}

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


@dataclass
class SalesMetrics:
    total_sales: float
    total_transactions: int
    avg_order_value: float
    total_refunds: float
    unique_members: int


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

        CREATE TABLE IF NOT EXISTS sales_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            branch_name TEXT,
            payment_datetime TEXT NOT NULL,
            payment_date TEXT,
            payment_week TEXT,
            payment_month TEXT,
            member_name TEXT,
            contact TEXT,
            sale_id TEXT,
            product TEXT,
            sale_status TEXT,
            sale_price REAL,
            payment_category TEXT,
            payment_method TEXT,
            payment_type TEXT,
            deduction REAL,
            payment_amount REAL,
            outstanding_amount REAL,
            refund_method TEXT,
            refund_amount REAL,
            net_amount REAL,
            sales_rep TEXT,
            memo TEXT,
            points REAL,
            uploaded_file TEXT,
            created_at TEXT NOT NULL
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
        cleaned = value.strip().replace(",", "").replace("%", "")
        cleaned = re.sub(r"[^\d\.\-]", "", cleaned)
        if cleaned in {"", "-", "."}:
            return None
        value = cleaned
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


def _parse_payment_datetime(value: Any) -> datetime | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, (int, float)) and not pd.isna(value):
        # Excel serial date support
        try:
            parsed = pd.to_datetime(value, unit="d", origin="1899-12-30", errors="coerce")
            if not pd.isna(parsed):
                return parsed.to_pydatetime()
        except Exception:
            pass

    if isinstance(value, str):
        cleaned = value.strip()
        cleaned = cleaned.replace("오전", "AM").replace("오후", "PM")
        parsed = pd.to_datetime(cleaned, errors="coerce")
    else:
        parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    if isinstance(parsed, pd.Timestamp):
        return parsed.to_pydatetime()
    return parsed


def _format_week(value: datetime) -> str:
    iso_year, iso_week, _ = value.isocalendar()
    return f"{iso_year}-W{iso_week:02d}"


def _load_dataframe(file_path: Path) -> pd.DataFrame:
    return _load_dataframe_with_header(file_path, header=0)


def _load_dataframe_with_header(file_path: Path, header: int | None) -> pd.DataFrame:
    if file_path.suffix.lower() == ".csv":
        return pd.read_csv(file_path, header=header)
    return pd.read_excel(file_path, header=header)


def _build_column_map(
    df: pd.DataFrame,
    alias_map: dict[str, set[str]],
) -> dict[str, str]:
    df.columns = [str(col).strip() for col in df.columns]
    mapping: dict[str, str] = {}
    for col in df.columns:
        key = _normalize_key(col)
        for canonical, aliases in alias_map.items():
            if key in aliases:
                mapping[canonical] = col
                break
    return mapping


def _row_matches_expected(
    values: list[Any],
    alias_map: dict[str, set[str]],
    required: set[str],
) -> bool:
    found: set[str] = set()
    for value in values:
        key = _normalize_key(value)
        for canonical, aliases in alias_map.items():
            if key in aliases:
                found.add(canonical)
    return required.issubset(found)


def _find_header_row(
    file_path: Path,
    alias_map: dict[str, set[str]],
    required: set[str],
) -> int | None:
    try:
        probe = _load_dataframe_with_header(file_path, header=None)
    except Exception:
        return None

    for idx in range(min(8, len(probe.index))):
        row_values = probe.iloc[idx].tolist()
        if _row_matches_expected(row_values, alias_map, required):
            return idx

    return None


def _detect_schema(df: pd.DataFrame) -> tuple[str | None, dict[str, str]]:
    bodycodi_map = _build_column_map(df, NORMALIZED_BODYCODI_ALIASES)
    sales_map = _build_column_map(df, NORMALIZED_SALES_ALIASES)

    bodycodi_ok = BODYCODI_REQUIRED_COLUMNS.issubset(bodycodi_map)
    sales_ok = SALES_REQUIRED_COLUMNS.issubset(sales_map)

    if sales_ok and not bodycodi_ok:
        return "sales", sales_map
    if bodycodi_ok and not sales_ok:
        return "bodycodi", bodycodi_map
    if sales_ok:
        return "sales", sales_map
    if bodycodi_ok:
        return "bodycodi", bodycodi_map
    return None, {}


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

        schema, col_map = _detect_schema(df)
        if schema is None:
            header_row = _find_header_row(
                file_path,
                NORMALIZED_SALES_ALIASES,
                SALES_REQUIRED_COLUMNS,
            )
            if header_row is None:
                header_row = _find_header_row(
                    file_path,
                    NORMALIZED_BODYCODI_ALIASES,
                    BODYCODI_REQUIRED_COLUMNS,
                )

            if header_row is not None:
                try:
                    df = _load_dataframe_with_header(file_path, header=header_row)
                    schema, col_map = _detect_schema(df)
                except Exception:
                    schema = None
                    col_map = {}

        if schema is None:
            invalid_files.append(file_path.name)
            continue

        inserted_for_file = 0
        if schema == "bodycodi":
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

                branch_col = col_map.get("branch")
                branch = _safe_text(row[branch_col]) if branch_col else ""
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
        else:
            for _, row in df.iterrows():
                payment_dt = _parse_payment_datetime(row[col_map["payment_datetime"]])
                if not payment_dt:
                    skipped_rows += 1
                    continue

                payment_amount = _safe_float(row[col_map["payment_amount"]])
                if payment_amount is None:
                    skipped_rows += 1
                    continue

                refund_col = col_map.get("refund_amount")
                refund_amount = _safe_float(row[refund_col]) if refund_col else 0.0
                refund_amount = refund_amount if refund_amount is not None else 0.0
                net_amount = payment_amount - refund_amount

                branch_col = col_map.get("branch_name")
                member_col = col_map.get("member_name")
                contact_col = col_map.get("contact")
                sale_id_col = col_map.get("sale_id")
                product_col = col_map.get("product")
                status_col = col_map.get("sale_status")
                sale_price_col = col_map.get("sale_price")
                payment_category_col = col_map.get("payment_category")
                payment_method_col = col_map.get("payment_method")
                payment_type_col = col_map.get("payment_type")
                deduction_col = col_map.get("deduction")
                outstanding_col = col_map.get("outstanding_amount")
                refund_method_col = col_map.get("refund_method")
                sales_rep_col = col_map.get("sales_rep")
                memo_col = col_map.get("memo")
                points_col = col_map.get("points")

                db.execute(
                    """
                    INSERT INTO sales_records (
                        branch_name, payment_datetime, payment_date, payment_week, payment_month,
                        member_name, contact, sale_id, product, sale_status, sale_price,
                        payment_category, payment_method, payment_type, deduction, payment_amount,
                        outstanding_amount, refund_method, refund_amount, net_amount,
                        sales_rep, memo, points, uploaded_file, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        _safe_text(row[branch_col]) if branch_col else "",
                        payment_dt.isoformat(sep=" "),
                        payment_dt.date().isoformat(),
                        _format_week(payment_dt),
                        payment_dt.strftime("%Y-%m"),
                        _safe_text(row[member_col]) if member_col else "",
                        _safe_text(row[contact_col]) if contact_col else "",
                        _safe_text(row[sale_id_col]) if sale_id_col else "",
                        _safe_text(row[product_col]) if product_col else "",
                        _safe_text(row[status_col]) if status_col else "",
                        _safe_float(row[sale_price_col]) if sale_price_col else None,
                        _safe_text(row[payment_category_col]) if payment_category_col else "",
                        _safe_text(row[payment_method_col]) if payment_method_col else "",
                        _safe_text(row[payment_type_col]) if payment_type_col else "",
                        _safe_float(row[deduction_col]) if deduction_col else None,
                        payment_amount,
                        _safe_float(row[outstanding_col]) if outstanding_col else None,
                        _safe_text(row[refund_method_col]) if refund_method_col else "",
                        refund_amount,
                        net_amount,
                        _safe_text(row[sales_rep_col]) if sales_rep_col else "",
                        _safe_text(row[memo_col]) if memo_col else "",
                        _safe_float(row[points_col]) if points_col else None,
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


def compute_sales_metrics() -> SalesMetrics:
    db = get_db()
    row = db.execute(
        """
        SELECT
            SUM(net_amount) AS total_sales,
            COUNT(CASE WHEN net_amount > 0 THEN 1 END) AS total_transactions,
            SUM(refund_amount) AS total_refunds
        FROM sales_records
        """
    ).fetchone()

    total_sales = float(row["total_sales"] or 0)
    total_transactions = int(row["total_transactions"] or 0)
    total_refunds = float(row["total_refunds"] or 0)

    unique_members_row = db.execute(
        """
        SELECT COUNT(DISTINCT member_name) AS unique_members
        FROM sales_records
        WHERE member_name IS NOT NULL AND member_name != ''
        """
    ).fetchone()
    unique_members = int(unique_members_row["unique_members"] or 0)

    avg_order_value = round(
        total_sales / total_transactions, 2
    ) if total_transactions else 0.0

    return SalesMetrics(
        total_sales=round(total_sales, 2),
        total_transactions=total_transactions,
        avg_order_value=avg_order_value,
        total_refunds=round(total_refunds, 2),
        unique_members=unique_members,
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
            upload_files = request.files.getlist("data_files")
            folder_files = request.files.getlist("data_folder")
            files = [file for file in (upload_files + folder_files) if file and file.filename]

            if not files:
                flash("업로드할 파일 또는 폴더를 선택해주세요.")
            else:
                RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
                saved_paths = []
                rejected = 0

                for upload_file in files:
                    if not _allowed_file(upload_file.filename):
                        rejected += 1
                        continue

                    original_name = upload_file.filename
                    base_name = secure_filename(Path(original_name).name)
                    if "/" in original_name or "\\" in original_name:
                        path_hash = hashlib.sha256(original_name.encode()).hexdigest()[:8]
                        filename = f"{path_hash}_{base_name}"
                    else:
                        filename = base_name

                    saved_path = RAW_DATA_DIR / filename
                    upload_file.save(saved_path)
                    saved_paths.append(saved_path)

                if not saved_paths:
                    flash("지원하지 않는 파일 형식입니다. (.xlsx, .csv)")
                else:
                    ingest_result = ingest_excel_files(saved_paths)
                    if rejected:
                        flash(f"일부 파일은 제외되었습니다: {rejected}건")
                    flash("업로드 반영이 완료되었습니다.")

    metrics = compute_sales_metrics()

    db = get_db()
    monthly_rows = db.execute(
        """
        SELECT payment_month AS period,
               ROUND(SUM(net_amount),2) AS sales,
               ROUND(
                   SUM(net_amount) * 1.0 /
                   NULLIF(COUNT(CASE WHEN net_amount > 0 THEN 1 END), 0),
                   2
               ) AS aov
        FROM sales_records
        WHERE payment_month IS NOT NULL AND payment_month != ''
        GROUP BY payment_month
        ORDER BY payment_month
        """
    ).fetchall()

    weekly_rows = db.execute(
        """
        SELECT payment_week AS period,
               ROUND(SUM(net_amount),2) AS sales,
               ROUND(
                   SUM(net_amount) * 1.0 /
                   NULLIF(COUNT(CASE WHEN net_amount > 0 THEN 1 END), 0),
                   2
               ) AS aov
        FROM sales_records
        WHERE payment_week IS NOT NULL AND payment_week != ''
        GROUP BY payment_week
        ORDER BY payment_week
        """
    ).fetchall()

    monthly_labels = [row["period"] for row in monthly_rows]
    monthly_sales = [row["sales"] or 0 for row in monthly_rows]
    monthly_aov = [row["aov"] or 0 for row in monthly_rows]

    weekly_labels = [row["period"] for row in weekly_rows]
    weekly_sales = [row["sales"] or 0 for row in weekly_rows]
    weekly_aov = [row["aov"] or 0 for row in weekly_rows]

    recent_records = db.execute(
        """
        SELECT member_name, payment_date, net_amount
        FROM sales_records
        ORDER BY id DESC
        LIMIT 8
        """
    ).fetchall()

    return render_template(
        "dashboard.html",
        metrics=metrics,
        ingest_result=ingest_result,
        recent_records=recent_records,
        monthly_labels=json.dumps(monthly_labels, ensure_ascii=False),
        monthly_sales=json.dumps(monthly_sales),
        monthly_aov=json.dumps(monthly_aov),
        weekly_labels=json.dumps(weekly_labels, ensure_ascii=False),
        weekly_sales=json.dumps(weekly_sales),
        weekly_aov=json.dumps(weekly_aov),
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
