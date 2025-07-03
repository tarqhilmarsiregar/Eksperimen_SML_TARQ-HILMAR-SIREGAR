import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from joblib import dump

def preprocess_data(data_path, target_column, save_pipeline_path, save_header_path):
    # 1. Load data
    df = pd.read_csv(data_path)
    
    # 2. Drop kolom ID jika ada
    if 'Person ID' in df.columns:
        df = df.drop(columns=['Person ID'])

    # 3. Pisahkan kolom Blood Pressure jika ada
    if 'Blood Pressure' in df.columns:
        df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True)
        df['Systolic_BP'] = pd.to_numeric(df['Systolic_BP'])
        df['Diastolic_BP'] = pd.to_numeric(df['Diastolic_BP'])
        df = df.drop(columns=['Blood Pressure'])

    # 4. Encode target column dengan LabelEncoder
    le = LabelEncoder()
    df[target_column] = le.fit_transform(df[target_column])

    # 5. Identifikasi kolom numerik dan kategorikal
    numeric_features = ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level',
                        'Stress Level', 'Heart Rate', 'Daily Steps', 'Systolic_BP', 'Diastolic_BP']
    
    categorical_features = [col for col in df.select_dtypes(include='object').columns if col != target_column]

    # 6. Simpan nama-nama kolom fitur
    column_names = df.drop(columns=[target_column]).columns
    pd.DataFrame(columns=column_names).to_csv(save_header_path, index=False)
    print(f"Nama kolom berhasil disimpan ke: {save_header_path}")

    # 7. Pipeline numerik
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # 8. Pipeline kategorikal
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # 9. Gabungkan dalam ColumnTransformer
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    # 10. Split data
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 11. Fit-transform dan simpan pipeline
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    dump(preprocessor, save_pipeline_path)
    print(f"Pipeline berhasil disimpan ke: {save_pipeline_path}")

    return X_train, X_test, y_train, y_test