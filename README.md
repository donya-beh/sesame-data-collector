# SESAME Data Collector

An automated data extraction pipeline for the SESAME (Stop Educator Sexual Abuse, Misconduct, and Exploitation) database. Given a CSV of news article URLs, the tool extracts structured misconduct data fields and outputs a clean CSV ready for database ingestion.

## What It Does

The pipeline processes each article URL through four steps:

1. Fetches the article text and publication date
2. Uses Claude Sonnet 4.5 (via AWS Bedrock) to extract structured fields — offender name, age, gender, role, arrest date, conviction status, victim information, and more
3. Looks up the normalized school district name, city, state, and ZIP from local NCES data files
4. Searches the web for Teacher/Coach of the Year recognition

The output is a 20-column CSV with one row per article.

## How to Use

### Requirements

- Python 3.10+
- AWS credentials with Bedrock access
- NCES data files (`ccd_school_districts.csv` and `ccd_public_schools.csv`) placed in the project root

### Setup

```bash
pip install -r requirements.txt
cp .env.example .env  # fill in your AWS credentials
```

### Web App

```bash
python app.py
```

Open **http://localhost:5001** in your browser. Upload a CSV with a `url` column, click Run Pipeline, and download the results when complete. If any articles fail to fetch, you can paste the article text manually to process them.

### Command Line

```bash
python run_workflow.py --input data/urls.csv --output-dir data/
```
