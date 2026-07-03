import joblib
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
    )
import seaborn as sns
import sys
from src.data_ingestion import ingest
import json
import os


data = "data/predictive_maintenance.csv"
df = ingest(data)
#load model
try:
    rf_model = joblib.load("model/rf_predictive_maintenance.joblib")
    lr_model = joblib.load("model/lr_predictive_maintenance.joblib")
    st.success(f"Models loaded")
           
except Exception as e:
    raise FileExistsError("Model file does not exist...")

with open("results/metrics.json", "r") as f:
    metrics = json.load(f)
with open("results/tests.json", "r") as t:
    test_metrics = json.load(t)

    
#Charts
def plot_confusion(y_test, rf_ypred,labels=None):
    title = f"Actual vs Predicted Values"
    if labels is None:
        labels = np.unique(rf_ypred)
        
    cm = confusion_matrix(y_test, rf_ypred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=labels, yticklabels=labels)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    return fig

def bar_graph(metrics_json):
    
    data = {
            "Accuracy": metrics_json["Accuracy_Score"],
            "F1": metrics_json["f1_Score"],
            "Precision": metrics_json["Precision_Score"],
            "Recall": metrics_json["recall_score"],
        }
        
    models = list(data["Accuracy"].keys())
    metrics = list(data.keys())
        
    # One simple figure
    fig, ax = plt.subplots(figsize=(8, 5))
        
    x_positions = range(len(metrics))
    width = 0.35
        
    for i, model in enumerate(models):
        values = [data[m][model] for m in metrics]
        ax.bar([p + i * width for p in x_positions], values, width, label=model)
        
    ax.set_xticks([p + width / 2 for p in x_positions])
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title("Model Comparison")
        
    plt.tight_layout()
    return fig

def confustion_plot(y_test, X_test, rf_model, log_model):
    # Don't retrain — just predict
    y_predict = rf_model.predict(X_test)
    y_predic = log_model.predict(X_test)
    
    
    rf_cm = confusion_matrix(y_test, y_predict, labels=rf_model.classes_)
    log_cm = confusion_matrix(y_test, y_predic, labels=log_model.classes_)
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    ConfusionMatrixDisplay(rf_cm, display_labels=rf_model.classes_).plot(ax=ax[0], cmap="Blues")
    ax[0].set_title("Random Forest Classifier")
    ax[0].tick_params(axis='x', rotation=90)
    
    ConfusionMatrixDisplay(log_cm, display_labels=log_model.classes_).plot(ax=ax[1], cmap="Blues")
    ax[1].set_title("Logistic Regression")
    ax[1].tick_params(axis='x', rotation=90)
    
    plt.tight_layout()
    return fig



def home_page():
    st.title("Predictive Maintenance Dashboard Analytics")
    st.divider()  
    
    
    X_rtest =np.array([test_metrics["Test01"]["values"], test_metrics["Test02"]["values"], 
                test_metrics["Test03"]["values"], test_metrics["Test04"]["values"],
                test_metrics["Test05"]["values"],test_metrics["Test06"]["values"]])
    
    # 0:"Heat Dissipation Failure"
    # 1:"No Failure"
    # 2:"Overstrain Failure"
    # 3:"Power Failure"
    # 4:"Random Failures"
    # 5:"Tool Wear Failure"
    y_dtest = np.array([1, 3, 1, 2, 2, 1])
            
    if rf_model is not None:
        y_pred = rf_model.predict(X_rtest)

        labels = sorted(df["Failure Type"].unique())
        label_map = {label: i for i, label in enumerate(labels)}
        y_pred_int = np.array([label_map[p] for p in y_pred])
            
        reverse_map = {v: k for k, v in label_map.items()}
        y_test_str = np.array([reverse_map[y] for y in y_dtest])


    #Chart selection
    st.subheader("Metrics Charts")
    chart_type = st.selectbox("select a chart",
                            ["Actual vs Predicted Tests Heatmap", "Model Accuracy Bar graph", "Confusion Metrics(From training data)"])

    if chart_type == "Actual vs Predicted Tests Heatmap":
        
        st.pyplot(plot_confusion(y_test_str, y_pred))
        
        # Build  Custom Predictions table
        feature_cols = [
            'Type', 'Air temperature [K]', 'Process temperature [K]', 
            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'
        ]

        results_df = pd.DataFrame(X_rtest, columns=feature_cols)
        results_df.insert(0, 'Test', [f"Data {i+1}" for i in range(len(X_rtest))])
        results_df['Predicted'] = y_pred  # your string predictions

        # Display
        st.subheader("Custom Predictions with Random Forest Classifier")
        st.table(results_df)
        
        # Per-class metrics
        st.subheader("Per-Class Metrics")
        correct = sum(1 for a, b in zip(y_dtest, y_pred_int) if a == b)
        st.write(f"Correct: {correct}/{len(y_dtest)} = {correct/len(y_dtest):.2%}")
        
    elif chart_type == "Model Accuracy Bar graph":
        st.pyplot(bar_graph(metrics))
        st.subheader("Accuracy & Model Score")
        score_data = {
            "Model": list(metrics["Accuracy_Score"].keys()),
            "Accuracy": list(metrics["Accuracy_Score"].values()),
            "Model Score": list(metrics["model_score"].values())
        }
        score_df = pd.DataFrame(score_data)

        st.table(score_df.style.format({"Accuracy": "{:.4f}", "Model Score": "{:.4f}"}))


        st.subheader("Detailed Metrics")
        detail_data = {
            "Model": list(metrics["f1_Score"].keys()),
            "F1 Score": list(metrics["f1_Score"].values()),
            "Precision": list(metrics["Precision_Score"].values()),
            "Recall": list(metrics["recall_score"].values())
        }
        detail_df = pd.DataFrame(detail_data)
        st.table(detail_df.style.format({"F1 Score": "{:.4f}", 
                                         "Precision": "{:.4f}", 
                                         "Recall": "{:.4f}"}))
        
    elif chart_type == "Confusion Metrics(From training data)":
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder

        df_encoded = df.copy()

        le = LabelEncoder()
        df_encoded["Type"] = le.fit_transform(df_encoded["Type"])

        X = df_encoded[['Type', 'Air temperature [K]','Process temperature [K]', 
                    'Rotational speed [rpm]', 'Torque [Nm]','Tool wear [min]']]
        y = df_encoded["Failure Type"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        st.pyplot(confustion_plot(y_test, X_test,rf_model,lr_model))


    #Dataframe
    st.divider()
    st.subheader("Data Used")
    st.dataframe(df)

    #Metrics
    st.divider()
    st.subheader("Metrics:")
    st.metric(label="Total rows", value=len(df))

    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.write("X-labels Columns used for training models", test_metrics["Data_value_structure"])
    with m_col2:
        st.write("Y-label Column(Failure Type) used for training models", {"Failure Type":[i for i in df["Failure Type"].unique()]})

    st.write("Model Metrics Training Performance",metrics)
    st.write("Test Notes:",f"{"Notes"*50}")
    st.write("Model Test Metrics", test_metrics)
    st.write("Notes:",f"{"Notes "*50}")
 



 

#           Side navigation bar

with st.sidebar:
    selected = option_menu(
        menu_title="MENU",
        options=["Analytics","Test ML With Custom Values"],
        menu_icon=None,
        default_index=0
    )
    
if selected=="Analytics":
    home_page()

elif selected=="Test ML With Custom Values":
    st.title("Test ML With Custom Values")
    st.divider()
    
    with st.form("predict"):
        Type = st.number_input("equipment type(1:low, 2:medium, 3:high)")
        air_temp = st.number_input("Air Temperature(K)")
        process_temp = st.number_input("Process Temperature(K)")
        rot_speed = st.number_input("Rotational speed(rpm)")
        torque = st.number_input("Torque(Nm)")
        tool_wear = st.number_input("Tool Wear(min)")
        
        predict_btn = st.form_submit_button("Predict")
        
        user_data = np.array([Type,air_temp,process_temp,
                              rot_speed,torque,tool_wear]).reshape(1,-1)
        form_data = {"Type":Type,"Air_Temperature(K)":air_temp,
                     "Process_Temperature(K)":process_temp,
                     "Rotational_speed(rpm)":rot_speed,
                     "Torque(Nm)":torque, "Tool_Wear(min)":tool_wear}
    
    if predict_btn:
        user_ypred = rf_model.predict(user_data)
        st.header("RESULTS:")
        
        st.divider()
        st.write(f"Prediction: {user_ypred[0]}")     
        
        #Data table
        st.divider()
        st.subheader("Your data")
        st.table(form_data)
        
        st.divider()
        st.json(form_data) 
       
