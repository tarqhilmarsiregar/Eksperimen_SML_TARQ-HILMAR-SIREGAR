from sklearn.preprocessing import StandardScaler, LabelEncoder
from joblib import dump
import pandas as pd
import os

def preprocess_data(data, target_column, save_pipeline_path, final_csv_path):
    df = data.copy()

    # Pisah kolom tekanan darah jika ada
    if 'Blood Pressure' in df.columns:
        df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True)
        df['Systolic_BP'] = pd.to_numeric(df['Systolic_BP'])
        df['Diastolic_BP'] = pd.to_numeric(df['Diastolic_BP'])
        df = df.drop(columns=['Blood Pressure'])

    # Hapus kolom ID jika ada
    if 'Person ID' in df.columns:
        df = df.drop(columns=['Person ID'])

    # Hapus baris yang memiliki missing values
    df = df.dropna()

    # Encode semua kolom object termasuk target (jika perlu)
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # Identifikasi fitur numerik (semua selain object yang sudah di-encode)
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Buat scaler dan standarize seluruh fitur numerik (kecuali target)
    features_to_scale = [col for col in numeric_features if col != target_column]
    scaler = StandardScaler()
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

    # Simpan hasil akhir ke CSV
    os.makedirs(os.path.dirname(final_csv_path), exist_ok=True)
    df.to_csv(final_csv_path, index=False)
    print(f"Hasil preprocessing disimpan di: {final_csv_path}")

    # Simpan scaler
    dump(scaler, save_pipeline_path)
    print(f"Scaler disimpan di: {save_pipeline_path}")

    return df

# Contoh pemanggilan
data = pd.read_csv('dataset_raw/Sleep_health_and_lifestyle_dataset.csv')

df_final = preprocess_data(
    data,
    target_column='Sleep Disorder',
    save_pipeline_path='preprocessing/dataset_preprocessing/preprocessing.joblib',
    final_csv_path='preprocessing/dataset_preprocessing/final_dataset.csv'
)