import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import DATA_FILE, TARGET, DROP_FROM_FEATURES, TEST_SIZE, RANDOM_STATE


def load_dataset():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            "Place College_student_behaviourial_data.csv inside the data folder."
        )

    df = pd.read_csv(DATA_FILE)

    required = [TARGET]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required target column(s): {missing}")

    if df.empty:
        raise ValueError("The dataset is empty.")

    return df


def prepare_data(df):
    # Remove duplicate rows before splitting.
    df = df.drop_duplicates().copy()

    # Drop rows where the target is missing; a supervised model needs labels.
    df = df.dropna(subset=[TARGET])

    drop_existing = [c for c in DROP_FROM_FEATURES if c in df.columns]
    X = df.drop(columns=drop_existing)
    y = df[TARGET].astype(str).str.strip()

    if X.shape[1] == 0:
        raise ValueError("No predictor columns remain after exclusions.")

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    transformers = []
    if numeric_features:
        transformers.append(("numeric", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test, preprocessor, numeric_features, categorical_features
