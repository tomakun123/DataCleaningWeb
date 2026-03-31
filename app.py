#!/usr/bin/env python3
"""
app.py — Web interface for the data pipeline.

Usage:
    python app.py
    Then open http://localhost:5000
"""

import sys
import os
import io
import uuid
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

try:
    import pandas as pd
    from flask import (
        Flask, render_template, request, session,
        redirect, url_for, send_from_directory, abort
    )
    from pipeline import (
        load_file, build_schema, infer_col_type,
        detect_anomalies, build_charts, build_profiles, render_report
    )
except ImportError as e:
    print(f"\n[ERROR] Missing dependency: {e}")
    print("Run:  pip install -r requirements.txt\n")
    sys.exit(1)


app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24))

UPLOAD_DIR = Path(tempfile.mkdtemp())
REPORT_DIR = Path(tempfile.mkdtemp())


# ── Web-friendly clean (no interactive prompts) ───────────────────────────────

def clean_data_web(df: pd.DataFrame, schema: list, strategy: str):
    issues = []

    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    if removed:
        issues.append(f"Removed {removed} duplicate row(s).")

    for col in schema:
        name = col["name"]
        if name not in df.columns:
            continue
        if col["type"] == "text":
            df[name] = df[name].astype(str).str.strip()
            df[name] = df[name].replace("nan", pd.NA)
        if col["type"] == "date":
            try:
                df[name] = pd.to_datetime(df[name], infer_datetime_format=True, errors="coerce")
            except Exception:
                pass
        null_pct = df[name].isna().mean() * 100
        if null_pct > 50:
            issues.append(f"Column '{name}' has {null_pct:.1f}% null values.")

    null_cols = [c for c in df.columns if df[c].isna().any()]
    if null_cols and strategy != "keep":
        if strategy == "fill":
            for col_name in null_cols:
                if pd.api.types.is_numeric_dtype(df[col_name]):
                    df[col_name] = df[col_name].fillna(df[col_name].median())
                else:
                    mode = df[col_name].mode()
                    if not mode.empty:
                        df[col_name] = df[col_name].fillna(mode[0])
            issues.append("Nulls filled (median for numbers, mode for text).")
        elif strategy == "drop":
            before = len(df)
            df = df.dropna()
            issues.append(f"Dropped {before - len(df)} row(s) with null values.")

    return df, issues


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files or request.files["file"].filename == "":
        return render_template("index.html", error="Please select a file.")

    f = request.files["file"]
    ext = Path(f.filename).suffix.lower()
    if ext not in (".csv", ".xlsx", ".xls"):
        return render_template("index.html", error="Only .csv, .xlsx, and .xls files are supported.")

    file_id = uuid.uuid4().hex
    save_path = UPLOAD_DIR / f"{file_id}{ext}"
    f.save(str(save_path))

    session["file_id"] = file_id
    session["file_ext"] = ext
    session["filename"] = f.filename

    # Excel: show sheet picker if multiple sheets, otherwise convert directly
    if ext in (".xlsx", ".xls"):
        try:
            xl = pd.ExcelFile(str(save_path))
            sheet_names = xl.sheet_names
        except Exception as e:
            return render_template("index.html", error=f"Could not read Excel file: {e}")

        if len(sheet_names) == 1:
            return _convert_sheet_and_configure(save_path, file_id, sheet_names[0], f.filename)

        # Build previews (row × col) for each sheet
        previews = []
        for sheet in sheet_names:
            try:
                raw = xl.parse(sheet, header=None, nrows=20)
                non_empty_counts = raw.apply(
                    lambda row: row.dropna().astype(str).str.strip().ne("").sum(), axis=1
                )
                header_row = int(non_empty_counts.idxmax())
                preview_df = xl.parse(sheet, header=header_row, nrows=3)
                previews.append({
                    "name": sheet,
                    "cols": len(preview_df.columns),
                    "sample_cols": list(preview_df.columns[:5]),
                })
            except Exception:
                previews.append({"name": sheet, "cols": 0, "sample_cols": []})

        return render_template("select_sheet.html", filename=f.filename, previews=previews)

    # CSV: go straight to configure
    try:
        df = load_file(str(save_path))
        schema = build_schema(df)
        null_cols = [c for c in df.columns if df[c].isna().any()]
    except Exception as e:
        return render_template("index.html", error=f"Could not read file: {e}")

    return render_template(
        "configure.html",
        schema=schema,
        filename=f.filename,
        row_count=f"{len(df):,}",
        has_nulls=bool(null_cols),
        null_cols=null_cols,
    )


def _convert_sheet_and_configure(save_path, file_id, sheet_name, filename):
    """Convert a specific Excel sheet to CSV and render the configure page."""
    try:
        xl = pd.ExcelFile(str(save_path))
        raw = xl.parse(sheet_name, header=None, nrows=20)
        non_empty_counts = raw.apply(
            lambda row: row.dropna().astype(str).str.strip().ne("").sum(), axis=1
        )
        header_row = int(non_empty_counts.idxmax())
        df = xl.parse(sheet_name, header=header_row)
        csv_path = UPLOAD_DIR / f"{file_id}.csv"
        df.to_csv(str(csv_path), index=False)
        session["file_ext"] = ".csv"
    except Exception as e:
        return render_template("index.html", error=f"Could not read sheet: {e}")

    schema = build_schema(df)
    null_cols = [c for c in df.columns if df[c].isna().any()]
    return render_template(
        "configure.html",
        schema=schema,
        filename=filename,
        row_count=f"{len(df):,}",
        has_nulls=bool(null_cols),
        null_cols=null_cols,
    )


@app.route("/select-sheet", methods=["POST"])
def select_sheet():
    sheet_name = request.form.get("sheet_name")
    file_id = session.get("file_id")
    file_ext = session.get("file_ext")
    filename = session.get("filename", "data")

    if not file_id or not sheet_name:
        return redirect(url_for("index"))

    save_path = UPLOAD_DIR / f"{file_id}{file_ext}"
    if not save_path.exists():
        return render_template("index.html", error="Session expired — please upload again.")

    return _convert_sheet_and_configure(save_path, file_id, sheet_name, filename)


@app.route("/run", methods=["POST"])
def run():
    file_id = session.get("file_id")
    file_ext = session.get("file_ext")
    filename = session.get("filename", "data")

    if not file_id:
        return redirect(url_for("index"))

    save_path = UPLOAD_DIR / f"{file_id}{file_ext}"
    if not save_path.exists():
        return render_template("index.html", error="Session expired — please upload again.")

    kept_cols = request.form.getlist("columns")
    null_strategy = request.form.get("null_strategy", "keep")

    if not kept_cols:
        df_tmp = load_file(str(save_path))
        schema_tmp = build_schema(df_tmp)
        null_cols = [c for c in df_tmp.columns if df_tmp[c].isna().any()]
        return render_template(
            "configure.html",
            schema=schema_tmp,
            filename=filename,
            row_count=f"{len(df_tmp):,}",
            has_nulls=bool(null_cols),
            null_cols=null_cols,
            error="Please select at least one column.",
        )

    try:
        df_raw = load_file(str(save_path))
        schema = build_schema(df_raw)

        df = df_raw[kept_cols].copy()
        kept_schema = [c for c in schema if c["name"] in kept_cols]

        orig_rows = len(df)
        df, issues = clean_data_web(df, kept_schema, null_strategy)
        dupes_removed = (
            orig_rows - len(df)
            if any("duplicate" in i.lower() for i in issues)
            else 0
        )

        anomaly_df, anomaly_notes = detect_anomalies(df)
        charts = build_charts(df)

        summary = {
            "rows": f"{len(df):,}",
            "cols": len(df.columns),
            "duplicates": dupes_removed,
            "anomalies": len(anomaly_df),
        }

        html = render_report(df, filename, summary, issues, anomaly_df, anomaly_notes, charts)

        report_id = uuid.uuid4().hex
        report_path = REPORT_DIR / f"{report_id}.html"
        report_path.write_text(html, encoding="utf-8")

        cleaned_path = REPORT_DIR / f"{report_id}_cleaned.csv"
        df.to_csv(str(cleaned_path), index=False)

        session["report_id"] = report_id

        return redirect(url_for("report"))

    except Exception as e:
        return render_template("index.html", error=f"Pipeline error: {e}")


@app.route("/report")
def report():
    report_id = session.get("report_id")
    if not report_id:
        return redirect(url_for("index"))
    report_path = REPORT_DIR / f"{report_id}.html"
    if not report_path.exists():
        return redirect(url_for("index"))

    html = report_path.read_text(encoding="utf-8")

    # Inject a sticky toolbar for navigation and downloads
    toolbar = """
<div style="position:fixed;top:0;left:0;right:0;z-index:9999;background:#2d3436;
            padding:10px 24px;display:flex;gap:16px;align-items:center;
            box-shadow:0 2px 8px rgba(0,0,0,.4)">
  <span style="color:#b2bec3;font-size:0.85rem;flex:1">Data Pipeline Report</span>
  <a href="/" style="color:#74b9ff;text-decoration:none;font-size:0.85rem">&#8592; New Upload</a>
  <a href="/download/report" style="color:#74b9ff;text-decoration:none;font-size:0.85rem">&#8659; HTML Report</a>
  <a href="#" onclick="window._downloadCSV ? window._downloadCSV() : window.location.href='/download/csv'; return false;" style="color:#55efc4;text-decoration:none;font-size:0.85rem">&#8659; Cleaned CSV</a>
</div>
<div style="height:44px"></div>
"""
    html = html.replace("<body>", "<body>" + toolbar, 1)
    return html


@app.route("/download/report")
def download_report():
    report_id = session.get("report_id")
    if not report_id:
        abort(404)
    stem = Path(session.get("filename", "report")).stem
    return send_from_directory(
        str(REPORT_DIR), f"{report_id}.html",
        as_attachment=True, download_name=f"report_{stem}.html"
    )


@app.route("/download/csv")
def download_csv():
    report_id = session.get("report_id")
    if not report_id:
        abort(404)
    stem = Path(session.get("filename", "data")).stem
    csv_path = REPORT_DIR / f"{report_id}_cleaned.csv"

    cols_param = request.args.get("cols", "").strip()
    if cols_param:
        requested = [c.strip() for c in cols_param.split(",") if c.strip()]
        df = pd.read_csv(str(csv_path))
        valid_cols = [c for c in requested if c in df.columns]
        if valid_cols:
            df = df[valid_cols]
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        from flask import Response
        return Response(
            buf.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": f'attachment; filename="cleaned_{stem}.csv"'},
        )

    return send_from_directory(
        str(REPORT_DIR), f"{report_id}_cleaned.csv",
        as_attachment=True, download_name=f"cleaned_{stem}.csv"
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n  Data Pipeline — Web UI")
    print("  Open http://localhost:5000 in your browser\n")
    app.run(debug=True, port=5000)
