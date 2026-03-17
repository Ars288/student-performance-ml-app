import joblib
import os

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from xgboost import XGBClassifier


def train_models(X_train, y_train_reg, y_train_cls):

    models = {}

    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train_reg)

    models["linear_regression"] = lr_model

    # Logistic Regression
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train, y_train_cls)

    models["logistic_regression"] = log_model

    # Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )

    rf_model.fit(X_train, y_train_cls)

    models["random_forest"] = rf_model

    # XGBoost
    xgb_model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6
    )

    xgb_model.fit(X_train, y_train_cls)

    models["xgboost"] = xgb_model

    # ensure models folder exists
    os.makedirs("models", exist_ok=True)

    # save models
    joblib.dump(models, "models/trained_models.pkl")

    return models


def evaluate_regression(model, X_test, y_test):

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return mae, mse, r2


def evaluate_classification(model, X_test, y_test):

    predictions = model.predict(X_test)

    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="weighted")

    return acc, f1