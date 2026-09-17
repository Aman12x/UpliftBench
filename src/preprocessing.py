import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATA_URL = (
    "https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/"
    "criteo-research-uplift-v2.1.csv.gz"
)
DEFAULT_CSV = Path(__file__).parent.parent / "data" / "criteo-research-uplift-v2.1.csv.gz"
OUTCOMES = ("visit", "conversion")


def download_data(dest=DEFAULT_CSV):
    """Fetch Criteo Uplift v2.1 (311 MB compressed) if it is not already on disk."""
    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(DATA_URL, dest)
    return dest


def load_data(path=DEFAULT_CSV, sample_frac=None, seed=42):
    """
    Load the dataset with a stable row_id column. With sample_frac, only the
    sampled rows ever reach pandas, so a 10% sample fits in a few hundred MB.
    The sample is a hash of (row_id, seed): same rows on every machine and thread count.
    """
    import duckdb

    con = duckdb.connect()
    con.execute("SET preserve_insertion_order = true")
    where = ""
    if sample_frac is not None:
        where = f"WHERE hash(row_id, {int(seed)}) % 10000 < {int(round(sample_frac * 10000))}"
    query = f"""
        SELECT * FROM (
            SELECT row_number() OVER () - 1 AS row_id, *
            FROM read_csv_auto('{Path(path)}')
        ) {where}
        ORDER BY row_id
    """
    df = con.execute(query).df()
    con.close()
    return df


def sample_rows(df, sample_frac, seed=42):
    """The same hash sample as load_data, for a frame that is already in memory."""
    keys = pd.util.hash_pandas_object(
        pd.DataFrame({"row_id": df["row_id"].to_numpy(), "seed": seed}), index=False
    ).to_numpy()
    return df[keys % 10000 < int(round(sample_frac * 10000))]


def split_data(df, outcome="visit"):
    if outcome not in OUTCOMES:
        raise ValueError(f"outcome must be one of {OUTCOMES}, got {outcome!r}")
    X = df[[f"f{i}" for i in range(12)]]
    T = df["treatment"]
    y = df[outcome]
    X_train, X_test, T_train, T_test, y_train, y_test = train_test_split(
        X, T, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, T_train, T_test, y_train, y_test


def fit_propensity(X_train, T_train):
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, T_train)
    return lr


def build_features(X_train, X_test, lr):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    propensity_train = lr.predict_proba(X_train)[:, 1].reshape(-1, 1)
    propensity_test = lr.predict_proba(X_test)[:, 1].reshape(-1, 1)

    X_train_final = np.hstack([X_train_scaled, propensity_train])
    X_test_final = np.hstack([X_test_scaled, propensity_test])
    return X_train_final, X_test_final
