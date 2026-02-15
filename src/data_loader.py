import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------
# Project root and data paths
# ---------------------------

# Works both in scripts and notebooks
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
# ---------------------------
# Helper functions
# ---------------------------

def load_csv(filename: str) -> pd.DataFrame:
    """
    Load a CSV file from DATA_DIR. Raises FileNotFoundError if missing.
    """
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Please check the file location.")
    return pd.read_csv(path)


def load_fraud_data() -> pd.DataFrame:
    """Load e-commerce fraud dataset."""
    return load_csv("Fraud_Data.csv")


def load_ip_country() -> pd.DataFrame:
    """Load IP-to-country mapping dataset."""
    return load_csv("IpAddress_to_Country.csv")


def load_creditcard() -> pd.DataFrame:
    """Load credit card fraud dataset."""
    return load_csv("creditcard.csv")


def ip_to_int(ip: str) -> int:
    """Convert IP address string to integer. Returns None if invalid."""
    try:
        parts = list(map(int, ip.split(".")))
        return sum(part << (8 * (3 - i)) for i, part in enumerate(parts))
    except Exception:
        return None


def merge_geolocation(fraud_df: pd.DataFrame, ip_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge fraud transactions with country info using IP ranges.
    IPs not found are labeled as 'Unknown'.
    """
    # Convert IP to integer
    fraud_df["ip_int"] = fraud_df["ip_address"].apply(ip_to_int)

    # Sort IP ranges for lookup
    ip_df = ip_df.sort_values("lower_bound_ip_address")

    # Lookup function
    def find_country(ip):
        if ip is None:
            return "Unknown"
        match = ip_df[(ip_df.lower_bound_ip_address <= ip) &
                      (ip_df.upper_bound_ip_address >= ip)]
        return match["country"].values[0] if not match.empty else "Unknown"

    # Apply
    fraud_df["country"] = fraud_df["ip_int"].apply(find_country)
    return fraud_df
