#!/usr/bin/env python3
import os
import json
import requests
import pandas as pd
import random
from datetime import datetime
from dateutil import tz

OUTPUT_JSONL = "energy_dataset_multi.jsonl"
OUTPUT_JSON = "energy_dataset_multi.json"
NUM_NUMERIC_SAMPLES_PER_SOURCE = 200
NUM_EXPLANATION_SAMPLES = 200
NUM_TIP_SAMPLES = 200
NUM_APPLIANCE_COMPARISONS_PER_FILE = 50

EIA_API_KEY = os.environ.get("EIA_API_KEY", "")
EIA_SERIES_IDS = [
    "EBA.US48-ALL.D.H",
]

NREL_API_KEY = os.environ.get("NREL_API_KEY", "")
NREL_PV_WATTS = False
NREL_CONFIG = {
    "system_capacity": 4.0,
    "azimuth": 180,
    "tilt": 20,
    "lat": 40.0,
    "lon": -105.0,
    "dataset": "intl",
}

LOCAL_CSVS = []

EFFICIENCY_TIPS = [
    "Switch to LED lighting to reduce electricity use significantly compared with incandescent bulbs.",
    "Install a programmable or smart thermostat to reduce HVAC runtime when the house is unoccupied.",
    "Seal air leaks around doors and windows and add insulation to reduce heating and cooling losses.",
    "Unplug or switch off devices that draw standby power to avoid phantom loads.",
    "Wash clothes in cold water and run full loads to save on water heating and pipeline energy.",
    "Consider adding energy storage (battery) if you have solar to shift self-consumption to peak hours.",
    "Perform regular maintenance on HVAC units (clean filters, service compressor) to keep efficiency high.",
    "Upgrade to Energy Star-rated appliances to reduce long-term electricity consumption.",
    "Use a clothesline or drying rack to cut dryer energy use when weather allows.",
    "Schedule heavy loads (dishwasher, laundry) to run at off-peak hours if on time-of-use rates."
]

FLOW_EXPLANATIONS = [
    "Electricity flows from the grid or local generation into your service panel and then to individual branch circuits that feed appliances and outlets.",
    "Solar panels create DC power that first passes through an inverter to become AC for household use; excess AC can be exported to the grid or sent to batteries.",
    "A home battery charges when generation exceeds demand or during low-price periods, and discharges to supply loads during peak or outage periods.",
    "Heat pumps move heat rather than generate it, using electricity to transfer thermal energy between inside and outside the home.",
    "HVAC consumes a lot of power because it runs compressors and fans; its usage spikes high when the system cycles during extreme weather."
]

random.seed(42)

def safe_datetime_parse(series):
    if isinstance(series, pd.Series):
        return pd.to_datetime(series, errors="coerce")
    return pd.to_datetime(series, errors="coerce")

def normalize_load_df(df, datetime_col_candidates=None, numeric_prefix=None):
    if datetime_col_candidates is None:
        datetime_col_candidates = ["datetime", "timestamp", "time", "date", "Date", "ts"]
    dt_col = None
    for c in datetime_col_candidates:
        if c in df.columns:
            dt_col = c
            break
    if dt_col is None:
        for c in df.columns:
            try:
                parsed = pd.to_datetime(df[c], errors="coerce")
                if parsed.notna().sum() > 0:
                    dt_col = c
                    break
            except Exception:
                continue
    if dt_col is None:
        raise ValueError("No datetime-like column found in CSV")
    df["datetime"] = safe_datetime_parse(df[dt_col])
    df = df.dropna(subset=["datetime"])
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if len(numeric_cols) == 0:
        for c in df.columns:
            if c == "datetime":
                continue
            coerced = pd.to_numeric(df[c], errors="coerce")
            if coerced.notna().sum() > 0:
                df[c] = coerced
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found in CSV for energy values")
    df = df.sort_values("datetime")
    df = df.reset_index(drop=True)
    return df

def fetch_eia_series(series_id, api_key):
    if not api_key:
        raise ValueError("EIA API key not provided")
    url = f"https://api.eia.gov/v2/electricity/rto/region-data/data/"
    params = {"api_key": api_key, "series_id": series_id, "frequency": "hourly", "data": "value"}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()
    items = payload.get("response", {}).get("data", [])
    if len(items) == 0:
        raise ValueError(f"No data returned for EIA series {series_id}")
    df = pd.DataFrame(items)
    if "period" in df.columns:
        df["datetime"] = pd.to_datetime(df["period"], errors="coerce")
    if "value" in df.columns:
        df["usage"] = pd.to_numeric(df["value"], errors="coerce")
    else:
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if len(num_cols) >= 1:
            df["usage"] = df[num_cols[0]]
        else:
            raise ValueError("EIA response contains no numeric usage column")
    df = df.dropna(subset=["datetime", "usage"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df[["datetime", "usage"]]

def fetch_nrel_pvwatts(nrel_key, config):
    if not nrel_key:
        raise ValueError("NREL API key not provided")
    base = "https://developer.nrel.gov/api/pvwatts/v6.json"
    params = {
        "api_key": nrel_key,
        "system_capacity": config.get("system_capacity", 4.0),
        "azimuth": config.get("azimuth", 180),
        "tilt": config.get("tilt", 20),
        "lat": config.get("lat"),
        "lon": config.get("lon"),
        "dataset": config.get("dataset", "tmy2"),
        "timeframe": "hourly",
    }
    r = requests.get(base, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()
    hourly = payload.get("outputs", {}).get("ac", None)
    if hourly is None:
        raise ValueError("PVWatts did not return hourly 'ac' output in this response")
    raise NotImplementedError("PVWatts parsing not implemented in generic script; implement per response format.")

def load_local_csv(path):
    df = pd.read_csv(path)
    df = normalize_load_df(df)
    return df

def generate_point_questions_from_df(df, n=100, label_prefix=None):
    qas = []
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if "usage" in numeric_cols:
        numeric_cols.remove("usage")
        numeric_cols.insert(0, "usage")
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns to generate QAs")
    for _ in range(n):
        row = df.sample(1).iloc[0]
        dt = pd.to_datetime(row["datetime"])
        main_col = numeric_cols[0]
        val = float(row[main_col])
        instr = f"What was the energy usage on {dt.strftime('%B %d, %Y at %H:%M')}?"
        out = f"The energy usage was {val:.3f} (units as in the source)."
        qas.append({"instruction": instr, "output": out})
    if "usage" in df.columns:
        df["year"] = df["datetime"].dt.year
        for year in sorted(df["year"].unique()):
            subset = df[df["year"] == year]
            if len(subset) == 0:
                continue
            peak = subset.loc[subset["usage"].idxmax()]
            low = subset.loc[subset["usage"].idxmin()]
            instr_p = f"In {year}, when was the peak energy usage and how much was it?"
            out_p = f"The peak usage was {float(peak['usage']):.3f} on {pd.to_datetime(peak['datetime']).strftime('%B %d, %Y at %H:%M')}."
            instr_l = f"In {year}, when was the lowest energy usage and how much was it?"
            out_l = f"The lowest usage was {float(low['usage']):.3f} on {pd.to_datetime(low['datetime']).strftime('%B %d, %Y at %H:%M')}."
            qas.append({"instruction": instr_p, "output": out_p})
            qas.append({"instruction": instr_l, "output": out_l})
        df = df.drop(columns=["year"])
    df["month"] = df["datetime"].dt.to_period("M")
    monthly = df.groupby("month")[numeric_cols[0]].mean().reset_index()
    for _, r in monthly.sample(min(12, len(monthly))).iterrows():
        instr = f"What was the average usage in {r['month'].strftime('%B %Y')}?"
        out = f"The average usage for {r['month'].strftime('%B %Y')} was {float(r[numeric_cols[0]]):.3f} (same units as source)."
        qas.append({"instruction": instr, "output": out})
    df = df.drop(columns=["month"])
    if len(numeric_cols) > 1:
        comps = 0
        for _ in range(min(NUM_APPLIANCE_COMPARISONS_PER_FILE, n)):
            cand = df.dropna(subset=numeric_cols).sample(1).iloc[0]
            ts = pd.to_datetime(cand["datetime"])
            a, b = random.sample(numeric_cols, 2)
            av = float(cand[a])
            bv = float(cand[b])
            instr = f"At {ts.strftime('%B %d, %Y at %H:%M')}, which used more energy: {a} or {b}, and by how much?"
            out = f"{a} used more energy than {b} by {abs(av - bv):.3f} (values: {a}={av:.3f}, {b}={bv:.3f})."
            qas.append({"instruction": instr, "output": out})
            comps += 1
            if comps >= NUM_APPLIANCE_COMPARISONS_PER_FILE:
                break
    return qas

def generate_explanations(n=100):
    qas = []
    examples = [
        "Explain how energy flows through a home with solar panels and a battery.",
        "Describe how electricity travels from the grid to my refrigerator.",
        "Why is my air conditioner using much more energy in summer?",
        "How does a battery system help during peak demand?",
        "Explain the difference between energy generation and energy consumption in a home."
    ]
    for _ in range(n):
        instr = random.choice(examples)
        out = random.choice(FLOW_EXPLANATIONS)
        qas.append({"instruction": instr, "output": out})
    return qas

def generate_efficiency_tips(n=100):
    qas = []
    examples = [
        "How can I reduce my home's electricity bill?",
        "Give me one practical tip to improve home energy efficiency.",
        "What should I do to reduce HVAC energy usage?",
        "How can I optimize my home for solar self-consumption?",
    ]
    for _ in range(n):
        instr = random.choice(examples)
        out = random.choice(EFFICIENCY_TIPS)
        qas.append({"instruction": instr, "output": out})
    return qas

def generate_seasonal_insight_qas(df, n=40):
    qas = []
    df = df.copy()
    if "usage" not in df.columns:
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        main = numeric_cols[0]
        df["usage"] = df[main]
    df["month"] = df["datetime"].dt.month
    monthly = df.groupby("month")["usage"].mean()
    if len(monthly) < 3:
        return qas
    for _ in range(n):
        m1, m2 = random.sample(list(monthly.index), 2)
        v1 = monthly.loc[m1]
        v2 = monthly.loc[m2]
        instr = f"Compare typical energy usage between {datetime(2000, m1, 1).strftime('%B')} and {datetime(2000, m2, 1).strftime('%B')}."
        if v1 > v2:
            out = f"On average, {datetime(2000, m1, 1).strftime('%B')} has higher usage ({v1:.3f}) than {datetime(2000, m2, 1).strftime('%B')} ({v2:.3f}), likely due to seasonal heating/cooling needs."
        else:
            out = f"On average, {datetime(2000, m2, 1).strftime('%B')} has higher usage ({v2:.3f}) than {datetime(2000, m1, 1).strftime('%B')} ({v1:.3f}), likely due to seasonal heating/cooling needs."
        qas.append({"instruction": instr, "output": out})
    df.drop(columns=["month"], inplace=True)
    return qas

def write_jsonl(qas, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in qas:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def main():
    all_qas = []
    for sid in EIA_SERIES_IDS:
        try:
            if not EIA_API_KEY:
                print(f"Skipping EIA series {sid} because EIA_API_KEY not set.")
                continue
            print(f"Fetching EIA series {sid}...")
            eia_df = fetch_eia_series(sid, EIA_API_KEY)
            eia_df = eia_df.rename(columns={"usage": "usage"})
            numeric_qas = generate_point_questions_from_df(eia_df, n=NUM_NUMERIC_SAMPLES_PER_SOURCE)
            seasonal_qas = generate_seasonal_insight_qas(eia_df, n=20)
            all_qas.extend(numeric_qas)
            all_qas.extend(seasonal_qas)
            print(f"Added {len(numeric_qas)+len(seasonal_qas)} QAs from EIA series {sid}")
        except Exception as e:
            print(f"Warning: failed to fetch/parse EIA series {sid}: {e}")
    if NREL_PV_WATTS:
        try:
            print("Fetching NREL PVWatts (example)...")
            pv_df = fetch_nrel_pvwatts(NREL_API_KEY, NREL_CONFIG)
        except Exception as e:
            print(f"Warning: failed to fetch/parse NREL PVWatts: {e}")
    for csv_path in LOCAL_CSVS:
        if not os.path.exists(csv_path):
            print(f"Local CSV {csv_path} not found, skipping.")
            continue
        try:
            print(f"Loading local CSV {csv_path}...")
            df = load_local_csv(csv_path)
            qas = generate_point_questions_from_df(df, n=min(NUM_NUMERIC_SAMPLES_PER_SOURCE, max(50, len(df)//10)))
            seasonal_qas = generate_seasonal_insight_qas(df, n=10)
            all_qas.extend(qas)
            all_qas.extend(seasonal_qas)
            print(f"Added {len(qas)+len(seasonal_qas)} QAs from {csv_path}")
        except Exception as e:
            print(f"Warning: failed to load/process CSV {csv_path}: {e}")
    all_qas.extend(generate_explanations(n=NUM_EXPLANATION_SAMPLES))
    all_qas.extend(generate_efficiency_tips(n=NUM_TIP_SAMPLES))
    random.shuffle(all_qas)
    seen = set()
    deduped = []
    for item in all_qas:
        key = (item["instruction"].strip(), item["output"].strip())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    write_jsonl(deduped, OUTPUT_JSONL)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(deduped, f, indent=2, ensure_ascii=False)
    print(f"Done. Wrote {len(deduped)} QA pairs to {OUTPUT_JSONL} and {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
