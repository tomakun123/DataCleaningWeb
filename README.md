# csv-cleaner-web

A browser-based data cleaning tool built with Flask and pandas. Upload a CSV or Excel file, choose which columns to keep and how to handle missing values, and get back a cleaned file plus a full HTML report — no command line required.

## Features

- Upload `.csv`, `.xlsx`, or `.xls` files
- Select columns to keep and configure null-handling strategy (fill, drop, or keep)
- Auto-detects duplicates, anomalies, and high-null columns
- Generates column profiles and charts
- Download the cleaned CSV and HTML report directly from the browser

## Setup

```bash
pip install -r requirements.txt
python app.py
```

Then open [http://localhost:5000](http://localhost:5000).

## Stack

Python · Flask · pandas · matplotlib · seaborn · Jinja2
