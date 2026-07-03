from typing import List

def feature_eng(data:str)->List:
    df = data
    df["Type"].unique()
    df['Type'] = df["Type"].map({"L":1, "M":2, "H":3})    
        
    feature_cols = ['Type', 'Air temperature [K]','Process temperature [K]', 
                    'Rotational speed [rpm]', 'Torque [Nm]','Tool wear [min]']

    xfeatures = df[feature_cols]
    ylabel = df['Failure Type']
    
    return [xfeatures, ylabel]
