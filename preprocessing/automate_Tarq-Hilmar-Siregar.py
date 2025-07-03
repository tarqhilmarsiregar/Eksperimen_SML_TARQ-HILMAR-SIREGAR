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

    # Hapus baris yang memiliki missing values di target atau fitur
    df = df.dropna(subset=[target_column])  # Pastikan target tidak kosong
    df = df.dropna()  # Hapus baris yang memiliki NaN di fitur lainnya

    # Encode target jika berupa string
    if df[target_column].dtype == 'object':
        le_target = LabelEncoder()
        df[target_column] = le_target.fit_transform(df[target_column])

    # Identifikasi fitur numerik dan kategorikal
    numeric_features = df.select_dtypes(include=['int64', 'float64']).drop(columns=[target_column]).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()

    # Encode fitur kategorikal dengan LabelEncoder
    for col in categorical_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # Pisah fitur dan target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Standardisasi fitur numerik
    scaler = StandardScaler()
    X[numeric_features] = scaler.fit_transform(X[numeric_features])

    # Gabungkan kembali
    df_final = X.copy()
    df_final[target_column] = y.reset_index(drop=True)

    # Simpan sebagai CSV
    os.makedirs(os.path.dirname(final_csv_path), exist_ok=True)
    df_final.to_csv(final_csv_path, index=False)
    print(f"Hasil preprocessing disimpan di: {final_csv_path}")

    # Simpan scaler
    dump(scaler, save_pipeline_path)
    print(f"Scaler disimpan di: {save_pipeline_path}")

    return df_final

data = pd.read_csv('dataset_raw/Sleep_health_and_lifestyle_dataset.csv')

df_final = preprocess_data(
    data,
    target_column='Sleep Disorder',
    save_pipeline_path='preprocessing/dataset_preprocessing/preprocessing.joblib',
    final_csv_path='preprocessing/dataset_preprocessing/final_dataset.csv'
)