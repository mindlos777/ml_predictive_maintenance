import pandas as pd
import os

def ingest(file:str):
    try:
        df = pd.read_csv(file)
    except Exception as e:
        raise FileExistsError("Error: document not found")
    
    #Cleaning dataset with missing values  if available   
    if df.isnull().sum().sum() != 0:
        print("Dataset contains missing values")
        
        for col in df.columns:
            df[col].dropna(df[col].mode(), inplace=True)
            
        print("Handled missing values...")
    
    return df