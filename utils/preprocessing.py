import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def load_dataset(path="data/students.csv"):

    df = pd.read_csv(path)

    df.rename(columns={
        "race/ethnicity": "race_ethnicity"
    }, inplace=True)

    return df


def create_target_variable(df):

    df["average_score"] = (
        df["math_score"] +
        df["reading_score"] +
        df["writing_score"]
    ) / 3

    conditions = [
        df["average_score"] < 50,
        (df["average_score"] >= 50) & (df["average_score"] < 75),
        df["average_score"] >= 75
    ]

    categories = ["Low", "Medium", "High"]

    df["performance_category"] = pd.cut(
        df["average_score"],
        bins=[0, 50, 75, 100],
        labels=categories
    )

    # encode labels for ML models
    df["performance_category"] = df["performance_category"].map({
        "Low":0,
        "Medium":1,
        "High":2
    })

    return df


def encode_features(df):

    categorical_columns = [
        "gender",
        "race_ethnicity",
        "parental_level_of_education",
        "lunch",
        "test_preparation_course"
    ]

    label_encoders = {}

    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    return df, label_encoders


def prepare_training_data(df):

    X = df.drop(
        ["math_score", "performance_category"],
        axis=1
    )

    y_regression = df["math_score"]

    y_classification = df["performance_category"]

    return X, y_regression, y_classification


def split_dataset(X, y_reg, y_cls):

    X_train, X_test, y_train_reg, y_test_reg = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )

    X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
        X, y_cls, test_size=0.2, random_state=42
    )

    return (
        X_train,
        X_test,
        y_train_reg,
        y_test_reg,
        X_train_cls,
        X_test_cls,
        y_train_cls,
        y_test_cls
    )