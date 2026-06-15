# PREDIKSI INDEKS PEMBANGUNAN MANUSIA (IPM) SULAWESI TENGAH
# Menggunakan Regresi Linear dan Support Vector Classification
# Dataset: BPS Sulawesi Tengah 2019-2023
 
# 1. Import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
 
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC, SVR
 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, classification_report

# 2. Membaca dataset
# Jika dijalankan di LOKAL (Windows), gunakan path seperti contoh di bawah:
# file_path = r"D:\Tugas\Dataset_IPM_Sulawesi_Tengah_2019-2023.xlsx"
#
# Jika dijalankan di GOOGLE COLAB, upload file lalu gunakan:
# file_path = "/content/Dataset_IPM_Sulawesi_Tengah_2019-2023.xlsx"
 
file_path = "Dataset_IPM_Sulawesi_Tengah_2019-2023.xlsx"
 
# Data tabel utama berada di sheet "Data Utama", dengan header pada baris ke-3
df = pd.read_excel(file_path, sheet_name="Data Utama", header=2)
 
print("Data berhasil dibaca")
print(df.to_string())

# 3. Merapikan nama kolom
df.columns = df.columns.str.replace("\n", " ").str.strip().str.lower()
 
df = df.rename(columns={
    "kabupaten/kota": "kabupaten_kota",
    "uhh (tahun)": "uhh",
    "hls (tahun)": "hls",
    "rls (tahun)": "rls",
    "pengeluaran per kapita (ribu rp/thn)": "pengeluaran_per_kapita",
    "kategori ipm": "kategori_ipm",
})
 
print("\nNama kolom dataset setelah dirapikan:")
print(df.columns.tolist())

# 4. Membersihkan data
df = df.drop(columns=["no"], errors="ignore")
df = df.dropna(subset=["kabupaten_kota", "tahun", "ipm"])

# 5. Kategori IPM
print("\nJumlah data tiap kategori IPM:")
print(df["kategori_ipm"].value_counts())

# 6. Menentukan fitur/input
fitur = ["tahun", "kabupaten_kota", "uhh", "hls", "rls", "pengeluaran_per_kapita"]
 
X = df[fitur]

# Target regresi -> memprediksi nilai IPM (angka)
y_regresi = df["ipm"]
 
# Target klasifikasi SVC -> memprediksi kategori IPM (Sedang/Tinggi/Sangat Tinggi)
y_svc = df["kategori_ipm"]

# 7. Preprocessing data
preprocessor = ColumnTransformer(
    transformers=[
        ("angka", StandardScaler(), ["tahun", "uhh", "hls", "rls", "pengeluaran_per_kapita"]),
        ("kategori", OneHotEncoder(handle_unknown="ignore"), ["kabupaten_kota"])
    ]
)

# ============================================================
# MODEL 1: REGRESI LINEAR (memprediksi nilai IPM)
# ============================================================
 
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X, y_regresi, test_size=0.2, random_state=42
)
 
model_regresi = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ]
)
 
model_regresi.fit(X_train_reg, y_train_reg)
prediksi_regresi = model_regresi.predict(X_test_reg)
 
mae_lr = mean_absolute_error(y_test_reg, prediksi_regresi)
mse_lr = mean_squared_error(y_test_reg, prediksi_regresi)
rmse_lr = np.sqrt(mse_lr)
r2_lr = r2_score(y_test_reg, prediksi_regresi)
 
print("\n===================================================")
print("HASIL EVALUASI REGRESI LINEAR")
print("===================================================")
print(f"MAE  : {mae_lr:.2f}")
print(f"MSE  : {mse_lr:.2f}")
print(f"RMSE : {rmse_lr:.2f}")
print(f"R2   : {r2_lr:.4f}")

# ============================================================
# MODEL 2a: SUPPORT VECTOR CLASSIFICATION (memprediksi kategori IPM)
# ============================================================
 
X_train_svc, X_test_svc, y_train_svc, y_test_svc = train_test_split(
    X, y_svc, test_size=0.2, random_state=42, stratify=y_svc
)
 
model_svc = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", SVC(kernel="rbf"))
    ]
)
 
model_svc.fit(X_train_svc, y_train_svc)
prediksi_svc = model_svc.predict(X_test_svc)
 
akurasi = accuracy_score(y_test_svc, prediksi_svc)
 
print("\n===================================================")
print("HASIL EVALUASI SUPPORT VECTOR CLASSIFICATION")
print("===================================================")
print(f"Akurasi SVC: {akurasi:.4f}")
print("\nClassification Report:")
print(classification_report(y_test_svc, prediksi_svc))

# ============================================================
# MODEL 2b: SUPPORT VECTOR REGRESSION (untuk visualisasi scatter plot SVR)
# ============================================================
 
model_svr = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", SVR(kernel="rbf"))
    ]
)
 
model_svr.fit(X_train_reg, y_train_reg)
prediksi_svr = model_svr.predict(X_test_reg)
 
mae_svr = mean_absolute_error(y_test_reg, prediksi_svr)
mse_svr = mean_squared_error(y_test_reg, prediksi_svr)
rmse_svr = np.sqrt(mse_svr)
r2_svr = r2_score(y_test_reg, prediksi_svr)
 
print("\n===================================================")
print("HASIL EVALUASI SUPPORT VECTOR REGRESSION (SVR)")
print("===================================================")
print(f"MAE  : {mae_svr:.2f}")
print(f"MSE  : {mse_svr:.2f}")
print(f"RMSE : {rmse_svr:.2f}")
print(f"R2   : {r2_svr:.4f}")

# ============================================================
# MEMBUAT DATA ESTIMASI TAHUN 2026
# ============================================================
 
data_2026 = []
 
for kabupaten in df["kabupaten_kota"].unique():
    data_kab = df[df["kabupaten_kota"] == kabupaten]
    data_terakhir = data_kab[data_kab["tahun"].isin([2021, 2022, 2023])]
 
    data_2026.append({
        "tahun": 2026,
        "kabupaten_kota": kabupaten,
        "uhh": data_terakhir["uhh"].mean(),
        "hls": data_terakhir["hls"].mean(),
        "rls": data_terakhir["rls"].mean(),
        "pengeluaran_per_kapita": data_terakhir["pengeluaran_per_kapita"].mean()
    })
 
df_2026 = pd.DataFrame(data_2026)

# ============================================================
# PREDIKSI IPM DAN KATEGORI IPM TAHUN 2026
# ============================================================
 
X_2026 = df_2026[fitur]
 
df_2026["prediksi_ipm"] = model_regresi.predict(X_2026)
df_2026["kategori_prediksi"] = model_svc.predict(X_2026)
 
df_2026["uhh"] = df_2026["uhh"].round(2)
df_2026["hls"] = df_2026["hls"].round(2)
df_2026["rls"] = df_2026["rls"].round(2)
df_2026["pengeluaran_per_kapita"] = df_2026["pengeluaran_per_kapita"].round(2)
df_2026["prediksi_ipm"] = df_2026["prediksi_ipm"].round(2)
 
print("\n===================================================")
print("HASIL PREDIKSI IPM TAHUN 2026")
print("===================================================")
print(df_2026)