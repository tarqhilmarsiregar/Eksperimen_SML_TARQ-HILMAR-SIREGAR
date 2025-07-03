from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from joblib import dump
import pandas as pd
from numpy import savez_compressed

def preprocess_data(data, target_column, save_pipeline_path, save_header_path):
    df = data.copy()

    # Pisah kolom 'Blood Pressure' jika ada
    if 'Blood Pressure' in df.columns:
        df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True)
        df['Systolic_BP'] = pd.to_numeric(df['Systolic_BP'])
        df['Diastolic_BP'] = pd.to_numeric(df['Diastolic_BP'])
        df = df.drop(columns=['Blood Pressure'])

    # Hapus 'Person ID' jika ada
    if 'Person ID' in df.columns:
        df = df.drop(columns=['Person ID'])

    # Encode target jika string
    if df[target_column].dtype == 'object':
        le = LabelEncoder()
        df[target_column] = le.fit_transform(df[target_column])

    # Identifikasi fitur numerik dan kategorikal
    numeric_features = df.select_dtypes(include=['int64', 'float64']).drop(columns=[target_column]).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()

    # Simpan header kolom
    pd.DataFrame(columns=df.drop(columns=[target_column]).columns).to_csv(save_header_path, index=False)
    print(f"Header kolom disimpan ke: {save_header_path}")

    # Pipeline untuk fitur numerik
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Pipeline untuk fitur kategorikal
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Gabungkan pipeline dengan ColumnTransformer
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    # Split data
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Transformasi data
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    # Simpan pipeline
    dump(preprocessor, save_pipeline_path)
    print(f"Pipeline disimpan ke: {save_pipeline_path}")

    return X_train, X_test, y_train, y_test

# Muat dataset kamu
data = pd.read_csv('../dataset_raw/Sleep_health_and_lifestyle_dataset.csv')

# Panggil fungsi preprocessing
X_train, X_test, y_train, y_test = preprocess_data(
    data,                            # DataFrame asli kamu
    'Sleep Disorder',                # Kolom target
    'dataset_preprocessing/dataset_preprocessor_pipeline.joblib', # Lokasi untuk menyimpan pipeline
    'dataset_preprocessing/data.csv'               # Lokasi untuk menyimpan nama-nama kolom (tanpa target)
)

savez_compressed('dataset_preprocessing/processed_data.npz',
                 X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

print("Preprocessing selesai dan hasil disimpan.")