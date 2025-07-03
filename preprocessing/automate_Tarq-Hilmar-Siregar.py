from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from joblib import dump
import pandas as pd
import os

def preprocess_data(data, target_column, save_pipeline_path, final_csv_path):
    df = data.copy()

    # Pisah kolom 'Blood Pressure' jika ada
    if 'Blood Pressure' in df.columns:
        df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True)
        df['Systolic_BP'] = pd.to_numeric(df['Systolic_BP'])
        df['Diastolic_BP'] = pd.to_numeric(df['Diastolic_BP'])
        df = df.drop(columns=['Blood Pressure'])

    # Hapus kolom ID jika ada
    if 'Person ID' in df.columns:
        df = df.drop(columns=['Person ID'])

    # Encode target jika berupa string
    if df[target_column].dtype == 'object':
        le = LabelEncoder()
        df[target_column] = le.fit_transform(df[target_column])

    # Identifikasi fitur numerik dan kategorikal
    numeric_features = df.select_dtypes(include=['int64', 'float64']).drop(columns=[target_column]).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()

    # Pipeline untuk numerik
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Pipeline untuk kategorikal
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Gabungkan dalam ColumnTransformer
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    # Split data
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Transformasi
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # Ambil nama kolom hasil OneHotEncoder
    cat_encoded_cols = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_features)
    all_columns = numeric_features + cat_encoded_cols.tolist()

    # Gabungkan hasil ke dalam satu DataFrame
    df_train = pd.DataFrame(X_train_trans, columns=all_columns)
    df_train[target_column] = y_train.reset_index(drop=True)

    df_test = pd.DataFrame(X_test_trans, columns=all_columns)
    df_test[target_column] = y_test.reset_index(drop=True)

    df_final = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)

    # Pastikan folder output ada
    os.makedirs(os.path.dirname(final_csv_path), exist_ok=True)

    # Simpan dataset akhir ke CSV
    df_final.to_csv(final_csv_path, index=False)
    print(f"Dataset hasil preprocessing disimpan ke: {final_csv_path}")

    # Simpan pipeline
    dump(preprocessor, save_pipeline_path)
    print(f"Pipeline disimpan ke: {save_pipeline_path}")

    return df_final

import pandas as pd

# Muat dataset
data = pd.read_csv('dataset_raw/Sleep_health_and_lifestyle_dataset.csv')

# Jalankan preprocessing
df_final = preprocess_data(
    data,
    target_column='Sleep Disorder',
    save_pipeline_path='preprocessing/dataset_preprocessing/pipeline.joblib',
    final_csv_path='preprocessing/dataset_preprocessing/final_dataset.csv'
)