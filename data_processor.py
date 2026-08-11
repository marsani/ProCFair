import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

def load_and_preprocess(filepath, dataset_name='retail', sample_frac=1.0):
    print(f"Loading raw data for {dataset_name}...")
    
    if dataset_name == 'retail':
        df = pd.read_excel(filepath)
        if sample_frac < 1.0:
            df = df.sample(frac=sample_frac, random_state=42)

        df = df.dropna(subset=['Customer ID', 'InvoiceDate'])
        df = df[df['Quantity'] > 0]
        df = df[df['Price'] > 0]

        df['TotalSum'] = df['Quantity'] * df['Price']
        snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

        customer_data = df.groupby('Customer ID').agg({
            'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
            'Invoice': 'nunique',
            'Quantity': 'mean',
            'Price': 'mean',
            'TotalSum': 'sum',
            'Country': 'first' 
        }).rename(columns={
            'InvoiceDate': 'Recency',
            'Invoice': 'Frequency',
            'Quantity': 'AvgQuantity',
            'Price': 'AvgPrice',
            'TotalSum': 'Monetary'
        })

        customer_data['S'] = (customer_data['Country'] == 'United Kingdom').astype(int)
        median_monetary = customer_data['Monetary'].median()
        customer_data['Y'] = (customer_data['Monetary'] > median_monetary).astype(int)

        features = ['Recency', 'Frequency', 'AvgQuantity', 'AvgPrice']
        X = customer_data[features].values
        S = customer_data['S'].values
        Y = customer_data['Y'].values

    elif dataset_name == 'telco':
        df = pd.read_csv(filepath)
        if sample_frac < 1.0:
            df = df.sample(frac=sample_frac, random_state=42)
            
        # Basic cleaning
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df = df.dropna(subset=['TotalCharges'])
        
        # S: gender (Male = 1, Female = 0)
        df['S'] = (df['gender'] == 'Male').astype(int)
        
        # Y: Churn (Yes = 1, No = 0)
        df['Y'] = (df['Churn'] == 'Yes').astype(int)
        
        # Features X
        # For simplicity we use numeric features
        features = ['tenure', 'MonthlyCharges', 'TotalCharges']
        X = df[features].values
        S = df['S'].values
        Y = df['Y'].values

    elif dataset_name == 'credit':
        # Large dataset, strong sample frac recommended
        df = pd.read_csv(filepath)
        if sample_frac < 1.0:
            df = df.sample(frac=sample_frac, random_state=42)
            
        # Clean Age
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
        df = df[(df['Age'] > 0) & (df['Age'] < 100)]
        df = df.dropna(subset=['Age', 'Credit_Score'])
        
        # S: Age > 30 (1) else (0)
        df['S'] = (df['Age'] > 30).astype(int)
        
        # Y: Good credit score vs Poor/Standard
        df['Y'] = (df['Credit_Score'] == 'Good').astype(int)
        
        # Select numeric features
        df['Annual_Income'] = pd.to_numeric(df['Annual_Income'], errors='coerce')
        df['Outstanding_Debt'] = pd.to_numeric(df['Outstanding_Debt'], errors='coerce')
        df['Monthly_Balance'] = pd.to_numeric(df['Monthly_Balance'], errors='coerce')
        df = df.dropna(subset=['Annual_Income', 'Outstanding_Debt', 'Monthly_Balance'])
        
        features = ['Annual_Income', 'Outstanding_Debt', 'Monthly_Balance', 'Num_Bank_Accounts', 'Num_Credit_Card']
        X = df[features].values
        S = df['S'].values
        Y = df['Y'].values
        
    else:
        raise ValueError("Dataset not supported")

    # Scale X
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, S, Y, features

def inject_missing_data(X, missing_rate=0.2, mechanism='MCAR'):
    X_miss = X.copy()
    n_samples, n_features = X.shape
    
    if mechanism == 'MCAR':
        mask = np.random.rand(n_samples, n_features) < missing_rate
        X_miss[mask] = np.nan
    elif mechanism == 'MAR':
        for j in range(1, n_features): 
            p = 1 / (1 + np.exp(-X[:, 0])) 
            p = p * (missing_rate / p.mean())
            p = np.clip(p, 0, 1)
            mask = np.random.rand(n_samples) < p
            X_miss[mask, j] = np.nan
    return X_miss

def inject_missing_s(S, X, missing_rate_s=0.3):
    S_miss = S.astype(float).copy()
    n_samples = len(S)
    
    if missing_rate_s > 0:
        p = 1 / (1 + np.exp(-X[:, -1])) 
        p = p * (missing_rate_s / p.mean())
        p = np.clip(p, 0, 1)
        
        mask = np.random.rand(n_samples) < p
        S_miss[mask] = np.nan
        
    return S_miss
