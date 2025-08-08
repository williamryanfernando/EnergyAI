import requests
import pandas as pd
import json
import os
from datetime import datetime

OUTPUT_FILE = "energy_dataset.json"
USE_EIA = True  # If False, load from LOCAL_CSV
LOCAL_CSV = "sample_energy_data.csv"
EIA_API_KEY = "YOUR_EIA_API_KEY"
EIA_SERIES_ID = "EBA.US48-ALL.D.H"  
NUM_SAMPLES = 200

def fetch_eia_data(api_key, series_id):
    url = f"https://api.eia.gov/v2/electricity/rto/region-data/data/?api_key={api_key}&series_id={series_id}"
    r = requests.get(url)
    if r.status_code != 200:
        raise Exception(f"EIA API request failed: {r.status_code}")
    data = r.json()
    df = pd.DataFrame(data["response"]["data"])
    df["datetime"] = pd.to_datetime(df["period"])
    return df

def load_local_csv(path):
    df = pd.read_csv(path)
    if "datetime" not in df.columns:
        raise Exception("Local CSV must have a 'datetime' column")
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df

def generate_qa_pairs(df, num_samples=100):
    qa_pairs = []
    df = df.sort_values("datetime")
    usage_col = [c for c in df.columns if c not in ["datetime"]][0]

    for _ in range(num_samples):
        row = df.sample(1).iloc[0]
        dt = row["datetime"]
        usage = row[usage_col]
        instr = f"What was the energy usage on {dt.strftime('%B %d, %Y at %H:%M')}?"
        out = f"The energy usage was {usage:.2f} megawatt-hours."
        qa_pairs.append({"instruction": instr, "output": out})

    for year in df["datetime"].dt.year.unique():
        yearly = df[df["datetime"].dt.year == year]
        peak_row = yearly.loc[yearly[usage_col].idxmax()]
        instr = f"In {year}, when was the peak energy usage and how much was it?"
        out = f"The peak usage was {peak_row[usage_col]:.2f} MWh on {peak_row['datetime'].strftime('%B %d, %Y at %H:%M')}."
        qa_pairs.append({"instruction": instr, "output": out})

    return qa_pairs

def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(data)} Q&A pairs to {path}")

def main():
    if USE_EIA:
        df = fetch_eia_data(EIA_API_KEY, EIA_SERIES_ID)
    else:
        df = load_local_csv(LOCAL_CSV)

    qa_pairs = generate_qa_pairs(df, NUM_SAMPLES)
    save_json(qa_pairs, OUTPUT_FILE)

if __name__ == "__main__":
    main()
