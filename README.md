# csv-cleaner-web

A browser-based data cleaning and profiling tool built with Python and Flask. Upload a CSV or Excel file, select which columns to retain, choose a strategy for handling missing values, and receive a cleaned dataset alongside a comprehensive HTML analytics report — all through a point-and-click interface with no command-line interaction required.

---

## Features

- Upload `.csv`, `.xlsx`, or `.xls` files
- Automatic detection of column data types (numeric, text, date, boolean)
- Interactive column selection with drag-to-reorder support
- Three missing-value strategies: fill (median/mode), drop rows, or keep as-is
- Automatic removal of exact duplicate rows
- Anomaly detection — statistical outliers (z-score > 3) and rare categorical values (< 0.5% frequency)
- Generated column profiles: null percentage, unique count, min/mean/max, most frequent value
- Distribution histograms and categorical bar charts embedded in the report
- Download the cleaned data as CSV, XLSX, or XLS
- Download the full report as a standalone HTML file or print to PDF via the browser

---

## Prerequisites

- Python 3.8 or later
- pip (included with Python)

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/<your-username>/csv-cleaner-web.git
cd csv-cleaner-web
```

**2. (Optional) Create and activate a virtual environment**

```bash
python -m venv venv

# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Start the server**

```bash
python app.py
```

**5. Open the app**

Navigate to [http://localhost:5000](http://localhost:5000) in your browser.

---

## Usage

### Step 1 — Upload a File

On the landing page, either drag and drop your file onto the upload area or click it to open a file picker. Supported formats are `.csv`, `.xlsx`, and `.xls`. Once a file is selected, click **Upload** to proceed.

### Step 2 — Select a Sheet (Excel files only)

If your Excel workbook contains multiple sheets, you will be prompted to choose which one to process. Select the appropriate sheet and click **Continue**. This step is skipped automatically for single-sheet workbooks and CSV files.

### Step 3 — Configure Columns and Null Handling

The configuration page presents a table with one row per column detected in your file. For each column you will see its inferred data type, the percentage of missing values, the number of unique values, and sample entries.

- **Select columns to keep** — Check the box next to each column you want to retain in the cleaned output. Uncheck any columns you wish to discard.
- **Reorder columns** — Drag rows by their handle to rearrange the column order. This order will be reflected in both the report and the downloaded file.
- **Choose a null-handling strategy** — Select one of the three options that applies globally to all retained columns:
  - **Fill** — Replace missing values with the column median (numeric columns) or the most frequent value (text columns).
  - **Drop** — Remove any row that contains at least one missing value.
  - **Keep** — Leave missing values as-is without modification.

Click **Run** when you are satisfied with your configuration.

### Step 4 — Review the Report

The report page displays a summary of all cleaning actions taken, followed by detailed analytics:

- **Summary bar** — Total rows retained, columns kept, duplicate rows removed, and anomalies detected.
- **Column profiles table** — One row per retained column showing type, non-null percentage, unique count, statistical range, and most frequent value. Columns can be reordered here as well.
- **Anomaly details** — Lists any statistical outliers (extreme numeric values) and rare categorical entries found in the cleaned data.
- **Distribution charts** — Histograms for up to eight numeric columns.
- **Categorical charts** — Bar charts showing the top 10 most frequent values for up to four text columns.

A sticky toolbar at the top of the report provides quick access to all download options.

### Step 5 — Download Your Results

From the sticky toolbar, choose any of the following:

| Option | Description |
|---|---|
| Download Report | Saves the full HTML report as a standalone file |
| Download CSV | Saves the cleaned data as a comma-separated values file |
| Download XLSX | Saves the cleaned data as a modern Excel workbook |
| Download XLS | Saves the cleaned data as a legacy Excel workbook |
| Print to PDF | Opens the browser print dialog; select "Save as PDF" |

---

## Tech Stack

| Component | Technology |
|---|---|
| Web framework | Flask |
| Data processing | pandas, numpy |
| Statistical analysis | scipy |
| Visualization | matplotlib, seaborn |
| Templating | Jinja2 |
| Excel I/O | openpyxl, xlwt |
| Production server | gunicorn |
