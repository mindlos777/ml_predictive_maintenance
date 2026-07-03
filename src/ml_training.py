import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
    )
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from feature_engineering import feature_eng
from data_ingestion import ingest
from typing import Dict
import joblib
import json
import os


def ml_train(data):
    x_values = data[0]
    y_labels = data[1]
    
    X_train, X_test, y_train, y_test = train_test_split(x_values, y_labels, test_size=0.25,train_size=0.75)
    
    def build_model(model, xtrain, ytrain):
        model.fit(xtrain, ytrain)
        return model
    
    #build models
    lr = LogisticRegression(random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    try:
        lr_cls = build_model(lr, X_train, y_train)
        rf_cls = build_model(rf, X_train, y_train)
    except Exception as e:
        print(f"Error training model(s): {e}")
        raise
    
    #predict
    lr_ypred = lr_cls.predict(X_test)
    rf_ypred = rf_cls.predict(X_test) 
    
    def acc_score():
        #Accuracy score & Classification score
        model_score = {"Logistic Regression":float(lr_cls.score(X_test,y_test)),
                        "Random Forest Classifier":float(rf_cls.score(X_test, y_test))}
        
        accuracy_report = {"Logistic Regression":float(accuracy_score(y_test, lr_ypred)),
                          "Random Forest Classifier":float(accuracy_score(y_test, rf_ypred))}
        
        f1_score_report = {"Logistic Regression":float(f1_score(y_test, lr_ypred, average='weighted')),
                           "Random Forest Classifier":float(f1_score(y_test, rf_ypred, average='weighted'))}
        
        recall_score_report = {"Logistic Regression":float(recall_score(y_test, lr_ypred, average='weighted')),
                               "Random Forest Classifier":float(recall_score(y_test, rf_ypred, average='weighted'))}

        precision_report = {"Logistic Regression":float(precision_score(y_test, lr_ypred, average='weighted')), 
                            "Random Forest Classifier":float(precision_score(y_test, rf_ypred, average='weighted'))}
        
        #conf_matrix_report = {"Logistic Regression":confusion_matrix(X_test, lr_ypred).tolist(), 
        #                      "Random Forest Classifier":confusion_matrix(X_test, rf_ypred).tolist()}
 
        metrics = {"model_score":model_score,
                   "Accuracy_Score":accuracy_report, 
                   "f1_Score":f1_score_report,
                   "Precision_Score":precision_report,
                   "recall_score": recall_score_report,
                   #"Confusion_metrics":conf_matrix_report
                   }
        
        return metrics
    
    #Tests(model prediction + accuracy)
    def tests():
        #Testing with Random Forest Classifier only 
        custom_data = np.array([
            [3, 298.8, 308.9, 1455, 41.3, 208],
            [2, 150, 397.9, 290, 100, 196],
            [1, 80, 200, 150, 50, 100],
            [2, 200, 500, 300, 150, 250],
            [1,298.4,308.2,1282,60.7,216],
            [3,198.4,389,2882,40.5,348]
        ])

        y_true_strings = y_labels.unique()

        str_labels = {}
        for i in range(0,len(y_true_strings)):
            str_labels[i]=y_true_strings[i]

        y_preds = rf_cls.predict(custom_data)
        y_pred_strings = [p for p in y_preds]
        
        #Generate report
        report = {"Data_value_structure":[i for i in x_values.columns],
                "Model_Accuracy":float(rf_cls.score(custom_data, y_pred_strings))
                }
        for d in range(0,len(custom_data)):
            report[f"Test0{d+1}"] = {"values":list(custom_data[d]),"prediction":y_pred_strings[d]}
            
        return report
        
    #Create directory and write to metrics.json and test.json
    output_dir = "results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            with open(os.path.join(output_dir, "metrics.json"), "w") as f:
                json.dump(acc_score(), f)
                print("\tLoaded metrics.json")
                
            with open(os.path.join(output_dir, "tests.json"), "w") as t:
                json.dump(tests(), t)
                print("\tLoaded tests.json")
                
        except Exception as e:
            print(f"Error: {e}")
        
    
    # Export best model
    def export_model():
        model_dir = "model"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            try:
                joblib.dump(rf_cls, os.path.join(model_dir,"rf_predictive_maintenance.joblib"))
                print(f"Model created successfully...\nModel saved as rf_predictive_maintenance.joblib in `{model_dir}` directory.")
            
            except Exception as e:
                print(f"Error : {e}")
                raise
        else:
            print("\tModel already exists...")
        
    return export_model()
      

if __name__ == "__main__":
    data = "data/predictive_maintenance.csv"
    f_eng = feature_eng(ingest(data))
    
    ml_train(f_eng)
    
    