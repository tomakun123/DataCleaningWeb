#!/usr/bin/env python3
"""
pipeline.py — Data Pipeline CLI Tool

Usage:
    python pipeline.py <file.csv>
    python pipeline.py <file.xlsx>
"""

import sys
import os
import io
import base64
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Imports ──────────────────────────────────────────────────────────────────

try:
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    from InquirerPy import inquirer
    from InquirerPy.base.control import Choice
    from jinja2 import Template
except ImportError as e:
    print(f"\n[ERROR] Missing dependency: {e}")
    print("Run:  pip install -r requirements.txt\n")
    sys.exit(1)


# ── Load File ─────────────────────────────────────────────────────────────────

def load_file(path: str) -> pd.DataFrame:
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        print(f"[ERROR] Unsupported file type: {ext}  (use .csv, .xlsx, or .xls)")
        sys.exit(1)
    return df


# ── Schema Inference ──────────────────────────────────────────────────────────

def infer_col_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    # Try date parsing on a sample
    sample = series.dropna().astype(str).head(20)
    try:
        pd.to_datetime(sample)
        return "date"
    except Exception:
        pass
    return "text"


def build_schema(df: pd.DataFrame) -> list[dict]:
    schema = []
    for col in df.columns:
        s = df[col]
        null_pct = round(s.isna().mean() * 100, 1)
        unique = s.nunique()
        col_type = infer_col_type(s)
        samples = s.dropna().head(3).tolist()
        schema.append({
            "name": col,
            "type": col_type,
            "null_pct": null_pct,
            "unique": unique,
            "samples": samples,
        })
    return schema


def print_schema(schema: list[dict]):
    print("\n" + "=" * 70)
    print(f"  {'COLUMN':<30} {'TYPE':<10} {'NULLS':>6}%  {'UNIQUE':>7}  SAMPLES")
    print("=" * 70)
    for col in schema:
        samples_str = ", ".join(str(v) for v in col["samples"][:3])
        if len(samples_str) > 25:
            samples_str = samples_str[:22] + "..."
        print(
            f"  {col['name']:<30} {col['type']:<10} {col['null_pct']:>6}%  "
            f"{col['unique']:>7}  {samples_str}"
        )
    print("=" * 70)


# ── Interactive Column Selector ───────────────────────────────────────────────

def select_columns(schema: list[dict]) -> list[str]:
    choices = [
        Choice(
            value=col["name"],
            name=f"{col['name']}  [{col['type']}, {col['null_pct']}% null]",
            enabled=True,
        )
        for col in schema
    ]
    selected = inquirer.checkbox(
        message="Select columns to KEEP (space to toggle, enter to confirm):",
        choices=choices,
        instruction="(space=toggle, a=all, i=invert, enter=confirm)",
    ).execute()

    if not selected:
        print("[ERROR] No columns selected. Exiting.")
        sys.exit(1)

    return selected


# ── Data Cleaning ─────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame, schema: list[dict]) -> tuple[pd.DataFrame, list[str], int]:
    issues = []

    # Remove exact duplicates
    before = len(df)
    df = df.drop_duplicates()
    dupes_removed = before - len(df)
    if dupes_removed:
        issues.append(f"Removed {dupes_removed} duplicate row(s).")

    for col in schema:
        name = col["name"]
        if name not in df.columns:
            continue

        # Strip whitespace from text columns
        if col["type"] == "text":
            df[name] = df[name].astype(str).str.strip()
            df[name] = df[name].replace("nan", pd.NA)

        # Parse dates
        if col["type"] == "date":
            try:
                df[name] = pd.to_datetime(df[name], errors="coerce")
            except Exception:
                pass

        # Warn on high nulls
        null_pct = df[name].isna().mean() * 100
        if null_pct > 50:
            issues.append(f"Column '{name}' has {null_pct:.1f}% null values.")

    # Null handling — ask user
    null_cols = [c for c in df.columns if df[c].isna().any()]
    if null_cols:
        print(f"\n  Columns with nulls: {', '.join(null_cols)}")
        strategy = inquirer.select(
            message="How should null values be handled?",
            choices=[
                Choice(value="fill", name="Fill — median for numbers, mode for text"),
                Choice(value="drop", name="Drop rows that contain any null"),
                Choice(value="keep", name="Keep as-is"),
            ],
        ).execute()

        if strategy == "fill":
            for col_name in null_cols:
                if pd.api.types.is_numeric_dtype(df[col_name]):
                    df[col_name] = df[col_name].fillna(df[col_name].median())
                else:
                    mode = df[col_name].mode()
                    if not mode.empty:
                        df[col_name] = df[col_name].fillna(mode[0])
            issues.append("Nulls filled (median/mode).")
        elif strategy == "drop":
            before = len(df)
            df = df.dropna()
            issues.append(f"Dropped {before - len(df)} row(s) with null values.")

    return df, issues, dupes_removed


# ── Anomaly Detection ─────────────────────────────────────────────────────────

def detect_anomalies(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    anomaly_flags = pd.Series([False] * len(df), index=df.index)
    anomaly_notes = []

    for col in df.select_dtypes(include="number").columns:
        col_clean = df[col].dropna()
        z = np.abs(stats.zscore(col_clean))
        outlier_idx = col_clean.index[z > 3]
        if len(outlier_idx):
            anomaly_flags[outlier_idx] = True
            anomaly_notes.append(f"'{col}': {len(outlier_idx)} outlier(s) (|z| > 3)")

    for col in df.select_dtypes(include="object").columns:
        counts = df[col].value_counts(normalize=True)
        rare = counts[counts < 0.005].index.tolist()
        if rare:
            rare_mask = df[col].isin(rare)
            anomaly_flags[rare_mask] = True
            anomaly_notes.append(
                f"'{col}': {len(rare)} rare value(s) (<0.5% frequency)"
            )

    anomaly_df = df[anomaly_flags].copy()
    return anomaly_df, anomaly_notes


# ── Chart Generation ──────────────────────────────────────────────────────────

def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=96)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def build_charts(df: pd.DataFrame) -> list[dict]:
    charts = []
    sns.set_theme(style="whitegrid", palette="muted")

    # Histograms for numeric columns
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    for col in numeric_cols[:8]:  # cap at 8 charts
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(df[col].dropna(), bins=30, color="#4C72B0", edgecolor="white")
        ax.set_title(f"Distribution: {col}", fontsize=11)
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        charts.append({"title": f"Distribution: {col}", "img": fig_to_base64(fig)})

    # Bar charts for categorical columns (top 10 values)
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    for col in cat_cols[:4]:  # cap at 4 charts
        top = df[col].value_counts().head(10)
        if top.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 3))
        top.plot(kind="barh", ax=ax, color="#55A868")
        ax.invert_yaxis()
        ax.set_title(f"Top Values: {col}", fontsize=11)
        ax.set_xlabel("Count")
        charts.append({"title": f"Top Values: {col}", "img": fig_to_base64(fig)})

    return charts


# ── HTML Report ───────────────────────────────────────────────────────────────

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Pipeline Report — {{ filename }}</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
           background: #f5f6fa; color: #2d3436; }
    header { background: #2d3436; color: white; padding: 24px 40px; }
    header h1 { font-size: 1.6rem; }
    header p  { color: #b2bec3; font-size: 0.9rem; margin-top: 4px; }
    main { max-width: 1100px; margin: 0 auto; padding: 32px 24px; }
    .card { background: white; border-radius: 8px; padding: 24px;
            margin-bottom: 24px; box-shadow: 0 1px 4px rgba(0,0,0,.08); }
    .card-header { display: flex; align-items: center; justify-content: space-between;
                   margin-bottom: 16px; border-bottom: 2px solid #dfe6e9; padding-bottom: 8px; }
    h2 { font-size: 1.1rem; }
    .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                 gap: 16px; }
    .stat { background: #f5f6fa; border-radius: 6px; padding: 16px; text-align: center; }
    .stat .val { font-size: 2rem; font-weight: 700; color: #0984e3; }
    .stat .lbl { font-size: 0.8rem; color: #636e72; margin-top: 4px; }
    table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
    th { background: #f5f6fa; text-align: left; padding: 8px 12px;
         border-bottom: 2px solid #dfe6e9; }
    td { padding: 7px 12px; border-bottom: 1px solid #f0f0f0; }
    tr:last-child td { border-bottom: none; }
    tr:hover td { background: #ffeaa7; }
    .chart-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
                  gap: 20px; }
    .chart-grid img { width: 100%; border-radius: 4px; }
    ul.issues { padding-left: 20px; }
    ul.issues li { margin-bottom: 6px; font-size: 0.9rem; }
    .badge { display: inline-block; padding: 2px 8px; border-radius: 12px;
             font-size: 0.78rem; font-weight: 600; }
    .badge-num  { background: #dfe6e9; color: #2d3436; }
    .badge-text { background: #ffeaa7; color: #6c5ce7; }
    .badge-date { background: #d1f2eb; color: #00b894; }
    .badge-bool { background: #fad7e1; color: #e17055; }
    .tag-warn { color: #d63031; font-weight: 600; }
    .empty { color: #b2bec3; font-style: italic; font-size: 0.9rem; }
    .copy-dropdown { position: relative; flex-shrink: 0; }
    .copy-btn {
      padding: 4px 12px;
      background: #f5f6fa;
      color: #636e72;
      border: 1px solid #dfe6e9;
      border-radius: 5px;
      font-size: 0.8rem;
      font-weight: 600;
      cursor: pointer;
      transition: background .15s, color .15s, border-color .15s;
      white-space: nowrap;
    }
    .copy-btn:hover { background: #dfe6e9; color: #2d3436; }
    .copy-btn.copied { background: #00b894; color: white; border-color: #00b894; }
    .copy-menu {
      display: none;
      position: absolute;
      right: 0;
      top: calc(100% + 6px);
      background: white;
      border: 1px solid #dfe6e9;
      border-radius: 7px;
      box-shadow: 0 4px 16px rgba(0,0,0,.12);
      z-index: 100;
      min-width: 190px;
      overflow: hidden;
    }
    .copy-menu.open { display: block; }
    .copy-menu button {
      display: flex;
      align-items: center;
      justify-content: space-between;
      width: 100%;
      padding: 9px 14px;
      background: none;
      border: none;
      cursor: pointer;
      font-size: 0.85rem;
      font-weight: 600;
      color: #2d3436;
      text-align: left;
      gap: 12px;
      transition: background .1s;
    }
    .copy-menu button:hover { background: #f5f6fa; }
    .copy-menu button span {
      font-weight: 400;
      color: #b2bec3;
      font-size: 0.78rem;
    }
    .copy-menu hr { border: none; border-top: 1px solid #f0f0f0; margin: 0; }
    /* ── Drag-to-reorder ──────────────────────────────────────────────────── */
    .drag-col { width: 28px; cursor: grab; user-select: none; text-align: center; }
    .drag-col:active { cursor: grabbing; }
    .drag-handle-icon {
      display: inline-block;
      width: 20px; height: 20px; line-height: 20px;
      background: #e8ecf0; border-radius: 4px;
      color: #636e72; font-size: 0.9rem; text-align: center;
      transition: background .15s, color .15s;
    }
    #profiles-table tbody tr:hover .drag-handle-icon { background: #0984e3; color: white; }
    #profiles-table tbody tr.row-dragging { opacity: 0.35; }
    #profiles-table tbody tr.row-drag-above td { box-shadow: inset 0 2px 0 #0984e3; }
    #profiles-table tbody tr.row-drag-below td { box-shadow: inset 0 -2px 0 #0984e3; }
    .drag-hint { font-size: 0.75rem; color: #b2bec3; margin-top: 2px; }
    #anomalies-table th {
      cursor: grab; user-select: none; position: relative; padding-bottom: 16px;
    }
    #anomalies-table th:active { cursor: grabbing; }
    #anomalies-table th::after {
      content: "⠿";
      position: absolute; bottom: 3px; left: 50%; transform: translateX(-50%);
      font-size: 0.65rem; color: #b2bec3; letter-spacing: 1px;
      transition: color .15s;
    }
    #anomalies-table th:hover::after { color: #0984e3; }
    #anomalies-table th.col-drag-source { opacity: 0.4; }
    #anomalies-table th.col-drag-over-left  { border-left:  3px solid #0984e3; }
    #anomalies-table th.col-drag-over-right { border-right: 3px solid #0984e3; }
    .col-reorder-hint {
      font-size: 0.75rem; color: #b2bec3;
      margin-bottom: 8px; text-align: right;
    }
  </style>
</head>
<body>
<header>
  <h1>Data Pipeline Report</h1>
  <p>File: {{ filename }} &nbsp;|&nbsp; Generated: {{ generated }}</p>
</header>
<main>

  <!-- Summary -->
  <div class="card">
    <div class="card-header"><h2>Summary</h2></div>
    <div class="stat-grid">
      <div class="stat"><div class="val">{{ summary.rows }}</div><div class="lbl">Rows</div></div>
      <div class="stat"><div class="val">{{ summary.cols }}</div><div class="lbl">Columns kept</div></div>
      <div class="stat"><div class="val">{{ summary.duplicates }}</div><div class="lbl">Duplicates removed</div></div>
      <div class="stat"><div class="val">{{ summary.anomalies }}</div><div class="lbl">Anomalies flagged</div></div>
    </div>
  </div>

  <!-- Column Profiles -->
  <div class="card">
    <div class="card-header">
      <div>
        <h2>Column Profiles</h2>
        <div class="drag-hint">&#8597; Drag rows to reorder &mdash; order affects CSV download</div>
      </div>
      <div class="copy-dropdown" id="dd-profiles">
        <button class="copy-btn" onclick="toggleDropdown('dd-profiles')">&#9113; Copy &#9662;</button>
        <div class="copy-menu">
          <button onclick="copyAs('profiles-table', 'tsv', 'dd-profiles')">TSV <span>Excel / Sheets paste</span></button>
          <button onclick="copyAs('profiles-table', 'csv', 'dd-profiles')">CSV <span>Comma-separated</span></button>
          <hr>
          <button onclick="copyAs('profiles-table', 'json', 'dd-profiles')">JSON <span>Array of objects</span></button>
          <button onclick="copyAs('profiles-table', 'markdown', 'dd-profiles')">Markdown <span>GitHub / docs table</span></button>
        </div>
      </div>
    </div>
    <table id="profiles-table">
      <thead>
        <tr>
          <th class="drag-col"></th>
          <th>Column</th><th>Type</th><th>Non-null</th><th>Unique</th>
          <th>Min / Mean / Max</th><th>Top Value</th>
        </tr>
      </thead>
      <tbody>
        {% for col in profiles %}
        <tr draggable="true">
          <td class="drag-col"><span class="drag-handle-icon">&#8597;</span></td>
          <td><strong>{{ col.name }}</strong></td>
          <td><span class="badge badge-{{ col.type }}">{{ col.type }}</span></td>
          <td>{{ col.non_null }}</td>
          <td>{{ col.unique }}</td>
          <td>{{ col.stats }}</td>
          <td>{{ col.top }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>

  <!-- Charts -->
  <div class="card">
    <div class="card-header"><h2>Charts</h2></div>
    {% if charts %}
    <div class="chart-grid">
      {% for chart in charts %}
      <div>
        <p style="font-size:0.85rem;color:#636e72;margin-bottom:6px">{{ chart.title }}</p>
        <img src="data:image/png;base64,{{ chart.img }}" alt="{{ chart.title }}" />
      </div>
      {% endfor %}
    </div>
    {% else %}
    <p class="empty">No numeric or categorical columns to chart.</p>
    {% endif %}
  </div>

  <!-- Anomalies -->
  <div class="card">
    <div class="card-header">
      <h2>Anomalies</h2>
      {% if anomaly_table %}
      <div class="copy-dropdown" id="dd-anomalies">
        <button class="copy-btn" onclick="toggleDropdown('dd-anomalies')">&#9113; Copy &#9662;</button>
        <div class="copy-menu">
          <button onclick="copyAs('anomalies-table', 'tsv', 'dd-anomalies')">TSV <span>Excel / Sheets paste</span></button>
          <button onclick="copyAs('anomalies-table', 'csv', 'dd-anomalies')">CSV <span>Comma-separated</span></button>
          <hr>
          <button onclick="copyAs('anomalies-table', 'json', 'dd-anomalies')">JSON <span>Array of objects</span></button>
          <button onclick="copyAs('anomalies-table', 'markdown', 'dd-anomalies')">Markdown <span>GitHub / docs table</span></button>
        </div>
      </div>
      {% endif %}
    </div>
    {% if anomaly_notes %}
    <ul class="issues" style="margin-bottom:16px">
      {% for note in anomaly_notes %}
      <li class="tag-warn">{{ note }}</li>
      {% endfor %}
    </ul>
    {% endif %}
    {% if anomaly_table %}
    <p class="col-reorder-hint">&#8596; Drag column headers left or right to reorder</p>
    <div style="overflow-x:auto">
    <table id="anomalies-table">
      <thead><tr>{% for h in anomaly_headers %}<th draggable="true">{{ h }}</th>{% endfor %}</tr></thead>
      <tbody>
        {% for row in anomaly_table %}
        <tr>{% for cell in row %}<td>{{ cell }}</td>{% endfor %}</tr>
        {% endfor %}
      </tbody>
    </table>
    </div>
    {% else %}
    <p class="empty">No anomalies detected.</p>
    {% endif %}
  </div>

  <!-- Data Quality Issues -->
  <div class="card">
    <div class="card-header"><h2>Data Quality Notes</h2></div>
    {% if issues %}
    <ul class="issues">
      {% for issue in issues %}
      <li>{{ issue }}</li>
      {% endfor %}
    </ul>
    {% else %}
    <p class="empty">No data quality issues found.</p>
    {% endif %}
  </div>

</main>
<script>
  // ── Dropdown toggle ────────────────────────────────────────────────────────
  function toggleDropdown(id) {
    const dd = document.getElementById(id);
    const menu = dd.querySelector('.copy-menu');
    const isOpen = menu.classList.contains('open');
    document.querySelectorAll('.copy-menu.open').forEach(m => m.classList.remove('open'));
    if (!isOpen) menu.classList.add('open');
  }
  document.addEventListener('click', (e) => {
    if (!e.target.closest('.copy-dropdown'))
      document.querySelectorAll('.copy-menu.open').forEach(m => m.classList.remove('open'));
  });

  // ── Format helpers ─────────────────────────────────────────────────────────
  // Skip drag-handle cells when reading table data
  function tableToData(tableId) {
    const table = document.getElementById(tableId);
    return Array.from(table.querySelectorAll('tr')).map(row =>
      Array.from(row.querySelectorAll('th, td'))
        .filter(cell => !cell.classList.contains('drag-col'))
        .map(cell => cell.innerText.trim())
    );
  }

  function toTSV(data) {
    return data.map(row => row.join('\\t')).join('\\n');
  }

  function toCSV(data) {
    return data.map(row =>
      row.map(cell => {
        if (cell.includes(',') || cell.includes('"') || cell.includes('\\n'))
          return '"' + cell.replace(/"/g, '""') + '"';
        return cell;
      }).join(',')
    ).join('\\n');
  }

  function toJSON(data) {
    const headers = data[0];
    const objects = data.slice(1).map(row => {
      const obj = {};
      headers.forEach((h, i) => { obj[h] = row[i] !== undefined ? row[i] : ''; });
      return obj;
    });
    return JSON.stringify(objects, null, 2);
  }

  function toMarkdown(data) {
    const header = '| ' + data[0].join(' | ') + ' |';
    const sep    = '| ' + data[0].map(() => '---').join(' | ') + ' |';
    const body   = data.slice(1).map(row => '| ' + row.join(' | ') + ' |');
    return [header, sep, ...body].join('\\n');
  }

  // ── Copy action ────────────────────────────────────────────────────────────
  function copyAs(tableId, format, dropdownId) {
    const data = tableToData(tableId);
    const text = { tsv: toTSV, csv: toCSV, json: toJSON, markdown: toMarkdown }[format](data);
    const dd   = document.getElementById(dropdownId);
    const btn  = dd.querySelector('.copy-btn');
    const menu = dd.querySelector('.copy-menu');
    const label = { tsv: 'TSV', csv: 'CSV', json: 'JSON', markdown: 'Markdown' }[format];

    function confirm() {
      menu.classList.remove('open');
      btn.textContent = '\\u2713 Copied as ' + label + '!';
      btn.classList.add('copied');
      setTimeout(() => { btn.innerHTML = '&#9113; Copy &#9662;'; btn.classList.remove('copied'); }, 1800);
    }
    navigator.clipboard.writeText(text).then(confirm).catch(() => {
      const ta = Object.assign(document.createElement('textarea'), { value: text });
      ta.style.cssText = 'position:fixed;opacity:0';
      document.body.appendChild(ta); ta.select(); document.execCommand('copy');
      document.body.removeChild(ta); confirm();
    });
  }

  // ── Column order (read from profiles table row order) ──────────────────────
  function getColumnOrder() {
    return Array.from(document.querySelectorAll('#profiles-table tbody tr')).map(row => {
      const strong = row.querySelector('td:not(.drag-col) strong');
      return strong ? strong.innerText.trim() : '';
    }).filter(Boolean);
  }

  // Called by the toolbar's "Cleaned CSV" button
  window._downloadCSV = function() {
    const cols = getColumnOrder();
    window.location.href = '/download/csv' +
      (cols.length ? '?cols=' + encodeURIComponent(cols.join(',')) : '');
  };

  // ── Profiles table: drag rows to reorder ──────────────────────────────────
  (function() {
    const tbody = document.querySelector('#profiles-table tbody');
    if (!tbody) return;
    let src = null;

    function clearRowHints() {
      tbody.querySelectorAll('tr').forEach(r =>
        r.classList.remove('row-drag-above', 'row-drag-below'));
    }

    tbody.addEventListener('dragstart', e => {
      src = e.target.closest('tr');
      if (!src) return;
      e.dataTransfer.effectAllowed = 'move';
      e.dataTransfer.setData('text/plain', '');
      setTimeout(() => src.classList.add('row-dragging'), 0);
    });
    tbody.addEventListener('dragover', e => {
      e.preventDefault();
      const tgt = e.target.closest('tr');
      if (!tgt || tgt === src) return;
      clearRowHints();
      const mid = tgt.getBoundingClientRect().top + tgt.getBoundingClientRect().height / 2;
      tgt.classList.add(e.clientY < mid ? 'row-drag-above' : 'row-drag-below');
    });
    tbody.addEventListener('dragleave', e => {
      if (!tbody.contains(e.relatedTarget)) clearRowHints();
    });
    tbody.addEventListener('drop', e => {
      e.preventDefault();
      const tgt = e.target.closest('tr');
      if (!tgt || tgt === src) { clearRowHints(); return; }
      const mid = tgt.getBoundingClientRect().top + tgt.getBoundingClientRect().height / 2;
      tbody.insertBefore(src, e.clientY < mid ? tgt : tgt.nextSibling);
      clearRowHints();
    });
    tbody.addEventListener('dragend', () => {
      if (src) src.classList.remove('row-dragging');
      clearRowHints();
      src = null;
    });
  })();

  // ── Anomalies table: drag column headers to reorder ───────────────────────
  (function() {
    const table = document.getElementById('anomalies-table');
    if (!table) return;
    const headerRow = table.querySelector('thead tr');
    if (!headerRow) return;
    let fromIdx = -1;

    function clearColHints() {
      headerRow.querySelectorAll('th').forEach(th =>
        th.classList.remove('col-drag-source', 'col-drag-over-left', 'col-drag-over-right'));
    }

    headerRow.addEventListener('dragstart', e => {
      const th = e.target.closest('th');
      if (!th) return;
      fromIdx = Array.from(headerRow.children).indexOf(th);
      e.dataTransfer.effectAllowed = 'move';
      e.dataTransfer.setData('text/plain', '');
      setTimeout(() => th.classList.add('col-drag-source'), 0);
    });
    headerRow.addEventListener('dragover', e => {
      e.preventDefault();
      const th = e.target.closest('th');
      if (!th) return;
      headerRow.querySelectorAll('th').forEach(t =>
        t.classList.remove('col-drag-over-left', 'col-drag-over-right'));
      const rect = th.getBoundingClientRect();
      th.classList.add(e.clientX < rect.left + rect.width / 2
        ? 'col-drag-over-left' : 'col-drag-over-right');
    });
    headerRow.addEventListener('dragleave', e => {
      if (!headerRow.contains(e.relatedTarget)) clearColHints();
    });
    headerRow.addEventListener('drop', e => {
      e.preventDefault();
      const th = e.target.closest('th');
      clearColHints();
      if (!th || fromIdx === -1) return;
      const rect = th.getBoundingClientRect();
      const dropBefore = e.clientX < rect.left + rect.width / 2;
      // Move every cell in fromIdx column to the drop position
      table.querySelectorAll('tr').forEach(row => {
        const cells = Array.from(row.children);
        const moving = cells[fromIdx];
        const ref    = cells[Array.from(headerRow.children).indexOf(th)];
        if (!moving || !ref || moving === ref) return;
        dropBefore ? ref.before(moving) : ref.after(moving);
      });
      fromIdx = -1;
    });
    headerRow.addEventListener('dragend', () => { clearColHints(); fromIdx = -1; });
  })();
</script>
</body>
</html>
"""


def build_profiles(df: pd.DataFrame) -> list[dict]:
    profiles = []
    for col in df.columns:
        s = df[col]
        col_type = infer_col_type(s)
        non_null = f"{s.notna().sum()} / {len(s)}"
        unique = s.nunique()

        if col_type == "numeric":
            mn = round(s.min(), 3)
            mean = round(s.mean(), 3)
            mx = round(s.max(), 3)
            stat_str = f"{mn} / {mean} / {mx}"
            top = "—"
        elif col_type == "date":
            stat_str = f"{s.min()} → {s.max()}" if s.notna().any() else "—"
            top = "—"
        else:
            stat_str = "—"
            vc = s.value_counts()
            top = str(vc.index[0]) if not vc.empty else "—"

        profiles.append({
            "name": col,
            "type": col_type,
            "non_null": non_null,
            "unique": unique,
            "stats": stat_str,
            "top": top,
        })
    return profiles


def render_report(
    df: pd.DataFrame,
    filename: str,
    summary: dict,
    issues: list[str],
    anomaly_df: pd.DataFrame,
    anomaly_notes: list[str],
    charts: list[dict],
) -> str:
    profiles = build_profiles(df)

    # Anomaly table (cap at 100 rows)
    anomaly_headers = list(anomaly_df.columns) if not anomaly_df.empty else []
    anomaly_table = (
        [list(row) for _, row in anomaly_df.head(100).iterrows()]
        if not anomaly_df.empty
        else []
    )

    tmpl = Template(HTML_TEMPLATE)
    return tmpl.render(
        filename=filename,
        generated=datetime.now().strftime("%Y-%m-%d %H:%M"),
        summary=summary,
        profiles=profiles,
        charts=charts,
        issues=issues,
        anomaly_notes=anomaly_notes,
        anomaly_headers=anomaly_headers,
        anomaly_table=anomaly_table,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        sys.exit(1)

    stem = Path(file_path).stem
    print(f"\n  Loading: {file_path}")

    # 1. Load
    df_raw = load_file(file_path)
    print(f"  {len(df_raw):,} rows × {len(df_raw.columns)} columns")
    print(df_raw.head(5).to_string(index=False))

    # 2. Schema
    schema = build_schema(df_raw)
    print_schema(schema)

    # 3. Column selection
    print()
    kept_cols = select_columns(schema)
    df = df_raw[kept_cols].copy()
    kept_schema = [c for c in schema if c["name"] in kept_cols]
    print(f"\n  Kept {len(kept_cols)} column(s): {', '.join(kept_cols)}")

    # 4. Clean
    df, issues, dupes_removed = clean_data(df, kept_schema)

    # 5. Anomalies
    anomaly_df, anomaly_notes = detect_anomalies(df)

    # 6. Charts
    print("\n  Building charts...")
    charts = build_charts(df)

    # 7. Summary
    summary = {
        "rows": f"{len(df):,}",
        "cols": len(df.columns),
        "duplicates": dupes_removed,
        "anomalies": len(anomaly_df),
    }

    # 8. Render report
    html = render_report(df, Path(file_path).name, summary, issues, anomaly_df, anomaly_notes, charts)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = Path(file_path).parent / f"report_{stem}_{ts}.html"
    report_path.write_text(html, encoding="utf-8")

    # 9. Save cleaned CSV
    cleaned_path = Path(file_path).parent / f"cleaned_{stem}.csv"
    df.to_csv(cleaned_path, index=False)

    # Done
    print(f"\n  {'─'*50}")
    print(f"  Done!")
    print(f"  HTML report : {report_path}")
    print(f"  Cleaned CSV : {cleaned_path}")
    print(f"  Rows        : {len(df):,}")
    print(f"  Columns     : {len(df.columns)}")
    print(f"  Anomalies   : {len(anomaly_df)}")
    print(f"  {'─'*50}\n")


if __name__ == "__main__":
    main()
