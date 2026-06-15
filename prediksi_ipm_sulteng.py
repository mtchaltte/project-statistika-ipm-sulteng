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