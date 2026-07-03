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
    model = joblib.load("model/rf_predictive_maintenance.joblib")
    st.success(f"Model loaded")
           
except Exception as e:
    raise FileExistsError("Model file does not exist...")

with open("results/metrics.json", "r") as f:
    metrics = json.load(f)
with open("results/tests.json", "r") as t:
    test_metrics = json.load(t)
    #values => np.array([type, air_temp, process_temp, rotation_speed, 
    #                    torque, tool_wear]).reshape(1, -1)
    #template(on click) => model.predict(values)
    
#Charts

def plot_confusion(y_test, rf_ypred,labels=None):
    title = f"Actual vs Predicted Values"
    if labels is None:
        labels = ["low", "medium", "high"]
        
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


def home_page():
    st.title("Predictive Maintenance Dashboard Analytics")
    st.divider()  
    
    
    X_rtest =np.array([test_metrics["Test01"]["values"], test_metrics["Test02"]["values"], 
                test_metrics["Test03"]["values"], test_metrics["Test04"]["values"],
                test_metrics["Test05"]["values"],test_metrics["Test06"]["values"]])
    y_test = np.array([1, 0, 2, 1, 0, 2])  # 0=low, 1=medium, 2=high
            
    if model is not None:
        y_pred = model.predict(X_rtest)

        labels = sorted(df["Failure Type"].unique())
        label_map = {label: i for i, label in enumerate(labels)}
        y_pred_int = np.array([label_map[p] for p in y_pred])
            
        reverse_map = {v: k for k, v in label_map.items()}
        y_test_str = np.array([reverse_map[y] for y in y_test])


    #Chart selection
    st.subheader("Metrics Charts")
    chart_type = st.selectbox("select a chart",
                            ["Confusion Metrics", "Model Accuracy Bar graph"])

    c_col1, c_col2= st.columns(2)
    if chart_type == "Confusion Metrics":
        with c_col1:
            st.pyplot(plot_confusion(y_test_str, y_pred))
        with c_col2:
            # Per-class metrics
            st.write("Per-Class Metrics")
            from sklearn.metrics import classification_report
            report = classification_report(y_test_str, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())
        
    elif chart_type == "Model Accuracy Bar graph":
        st.pyplot(bar_graph(metrics))

    #Dataframe
    st.subheader("Data")
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
        user_ypred = model.predict(user_data)
        st.header("RESULTS:")
        
        st.divider()
        st.write(f"Prediction: {user_ypred}")     
        
        #Data table
        st.divider()
        st.subheader("Your data")
        st.table(form_data)
        
        st.divider()
        st.json(form_data) 
       
