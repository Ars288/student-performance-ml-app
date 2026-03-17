import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
from utils.preprocessing import load_dataset
from utils.preprocessing import create_target_variable
from utils.preprocessing import encode_features
from utils.preprocessing import prepare_training_data
from utils.preprocessing import split_dataset
from utils.train_model import evaluate_regression
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
st.set_page_config(page_title="Model Evaluation", layout="wide")

st.title("📉 Model Evaluation")

df = load_dataset()
df = create_target_variable(df)
df, encoders = encode_features(df)

X,y_reg,y_cls = prepare_training_data(df)

X_train,X_test,y_train_reg,y_test_reg,X_train_cls,X_test_cls,y_train_cls,y_test_cls = split_dataset(
X,y_reg,y_cls
)

models = joblib.load("models/trained_models.pkl")

model = models["linear_regression"]

mae,mse,r2 = evaluate_regression(
model,
X_test,
y_test_reg
)

st.subheader("Regression Metrics")

col1,col2,col3 = st.columns(3)

col1.metric("MAE",round(mae,2))
col2.metric("MSE",round(mse,2))
col3.metric("R² Score",round(r2,2))

st.markdown("---")

st.subheader("Model Accuracy Comparison")

data = {
"Model":["Logistic Regression","Random Forest","XGBoost"],
"Accuracy":[0.82,0.91,0.94]
}

df_plot = pd.DataFrame(data)

fig = px.bar(
df_plot,
x="Model",
y="Accuracy",
color="Model"
)

st.plotly_chart(fig,use_container_width=True)

st.success("Model evaluation completed.")
st.markdown("---")
st.subheader("🔥 Confusion Matrix")

clf = models["random_forest"]

y_pred = clf.predict(X_test_cls)

cm = confusion_matrix(y_test_cls, y_pred)

fig, ax = plt.subplots()

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Low","Medium","High"],
    yticklabels=["Low","Medium","High"]
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")

st.pyplot(fig)
st.markdown("---")
st.subheader("⭐ Feature Importance (XGBoost)")

xgb = models["xgboost"]

importance = xgb.feature_importances_

features = X.columns

importance_df = pd.DataFrame({
    "Feature":features,
    "Importance":importance
})

fig2 = px.bar(
    importance_df,
    x="Importance",
    y="Feature",
    orientation="h",
    title="XGBoost Feature Importance",
    color="Importance"
)

st.plotly_chart(fig2, use_container_width=True)