"""
Automated ML Data Preprocessor
Week 5 Milestone

Input:
    customer_churn_dataset.csv

Output:
    X_train.csv
    X_test.csv
    y_train.csv
    y_test.csv

Techniques:
    - pandas.get_dummies() for categorical feature encoding
    - StandardScaler for numerical Z-score scaling
    - train_test_split(test_size=0.2, random_state=42)
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "customer_churn_dataset.csv"
OUTPUT_DIR = BASE_DIR / "processed"
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET = "Churn"


def main():
    df = pd.read_csv(INPUT_FILE)

    print("Original shape:", df.shape)

    # Drop identifier because it has no predictive meaning.
    if "CustomerID" in df.columns:
        df = df.drop(columns=["CustomerID"])

    # Separate features and target.
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Handle missing values.
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    for col in numeric_cols:
        X[col] = X[col].fillna(X[col].median())

    for col in categorical_cols:
        X[col] = X[col].fillna(X[col].mode()[0])

    # Convert target into a machine-readable binary label.
    y = y.map({"No": 0, "Yes": 1})

    # Encode categorical features using one-hot encoding.
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dtype=int)

    # Train-test split BEFORE scaling to avoid test-data leakage.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale only numerical columns using training data.
    scale_cols = [c for c in numeric_cols if c in X_train.columns]
    scaler = StandardScaler()

    X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_test[scale_cols] = scaler.transform(X_test[scale_cols])

    # Save required artifacts.
    X_train.to_csv(OUTPUT_DIR / "X_train.csv", index=False)
    X_test.to_csv(OUTPUT_DIR / "X_test.csv", index=False)
    y_train.to_csv(OUTPUT_DIR / "y_train.csv", index=False)
    y_test.to_csv(OUTPUT_DIR / "y_test.csv", index=False)

    # Save a preprocessing summary.
    summary = pd.DataFrame({
        "Metric": [
            "Original rows",
            "Original columns",
            "Processed feature columns",
            "Training rows",
            "Testing rows",
            "Target classes"
        ],
        "Value": [
            len(df),
            len(df.columns),
            X.shape[1],
            len(X_train),
            len(X_test),
            "0 = No, 1 = Yes"
        ]
    })
    summary.to_csv(OUTPUT_DIR / "preprocessing_summary.csv", index=False)

    print("\nPreprocessing completed successfully.")
    print("Processed feature shape:", X.shape)
    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test :", y_test.shape)
    print("\nFiles saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
