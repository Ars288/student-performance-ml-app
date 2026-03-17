from utils.preprocessing import load_dataset
from utils.preprocessing import create_target_variable
from utils.preprocessing import encode_features
from utils.preprocessing import prepare_training_data
from utils.preprocessing import split_dataset

from utils.train_model import train_models

print("Loading dataset...")

df = load_dataset()

print("Creating target variable...")

df = create_target_variable(df)

print("Encoding features...")

df, encoders = encode_features(df)

print("Preparing training data...")

X, y_reg, y_cls = prepare_training_data(df)

print("Splitting dataset...")

(
    X_train,
    X_test,
    y_train_reg,
    y_test_reg,
    X_train_cls,
    X_test_cls,
    y_train_cls,
    y_test_cls
) = split_dataset(X, y_reg, y_cls)

print("Training models...")

models = train_models(
    X_train,
    y_train_reg,
    y_train_cls
)

print("Models trained successfully!")

print("Saved to models/trained_models.pkl")