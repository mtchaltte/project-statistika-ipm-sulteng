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