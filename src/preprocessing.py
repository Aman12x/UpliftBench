import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(path):
    return pd.read_csv(path)


def split_data(df):
    X = df[[f"f{i}" for i in range(12)]]
    T = df["treatment"]
    y = df["visit"]
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
