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