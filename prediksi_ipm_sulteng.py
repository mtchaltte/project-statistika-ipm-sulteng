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

# ============================================================
# MENYIMPAN HASIL PREDIKSI KE EXCEL
# ============================================================
 
df_2026.to_excel("hasil_prediksi_ipm_2026.xlsx", index=False)
print("\nHasil prediksi berhasil disimpan ke file: hasil_prediksi_ipm_2026.xlsx")

# ============================================================
# VISUALISASI GRAFIK
# ============================================================
 
# Warna konsisten
WARNA_LR  = "#5B9BD5"   # biru
WARNA_SVR = "#ED7D31"   # oranye
 
fig = plt.figure(figsize=(14, 11))
fig.suptitle(
    "Prediksi IPM Sulawesi Tengah — Regresi Linear vs SVR",
    fontsize=15, fontweight="bold", y=0.98
)
 
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)
 
# ── Helper: garis ideal & batas sumbu ──────────────────────────────────────
def batas_sama(ax, x_data, y_data, margin=1.5):
    """Membuat skala sumbu x dan y sama dengan sedikit margin."""
    semua = np.concatenate([x_data, y_data])
    lo, hi = semua.min() - margin, semua.max() + margin
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    return lo, hi
 
def tambah_garis_ideal(ax, lo, hi):
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Ideal")
    ax.legend(fontsize=9)
 
# ── 1. Scatter Plot – Regresi Linear ───────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(y_test_reg, prediksi_regresi, color=WARNA_LR, alpha=0.75, edgecolors="white", s=70)
lo, hi = batas_sama(ax1, y_test_reg.values, prediksi_regresi)
tambah_garis_ideal(ax1, lo, hi)
ax1.set_xlabel("IPM Aktual", fontsize=10)
ax1.set_ylabel("IPM Prediksi", fontsize=10)
ax1.set_title(f"Regresi Linear\nR²={r2_lr:.4f}", fontsize=11)
ax1.grid(True, linestyle="--", alpha=0.4)
 
# ── 2. Scatter Plot – SVR ──────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(y_test_reg, prediksi_svr, color=WARNA_SVR, alpha=0.75, edgecolors="white", s=70)
lo, hi = batas_sama(ax2, y_test_reg.values, prediksi_svr)
tambah_garis_ideal(ax2, lo, hi)
ax2.set_xlabel("IPM Aktual", fontsize=10)
ax2.set_ylabel("IPM Prediksi", fontsize=10)
ax2.set_title(f"SVR\nR²={r2_svr:.4f}", fontsize=11)
ax2.grid(True, linestyle="--", alpha=0.4)
 
# ── 3. Bar Chart – Perbandingan Metrik ─────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
metrik_nama  = ["MAE", "RMSE", "R²"]
nilai_lr     = [mae_lr,  rmse_lr,  r2_lr]
nilai_svr    = [mae_svr, rmse_svr, r2_svr]
 
x = np.arange(len(metrik_nama))
lebar = 0.35
 
batang_lr  = ax3.bar(x - lebar/2, nilai_lr,  lebar, label="Regresi Linear", color=WARNA_LR,  alpha=0.85)
batang_svr = ax3.bar(x + lebar/2, nilai_svr, lebar, label="SVR",            color=WARNA_SVR, alpha=0.85)
 
# Label nilai di atas batang
for bar in batang_lr:
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8.5)
for bar in batang_svr:
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8.5)
 
ax3.set_xticks(x)
ax3.set_xticklabels(metrik_nama, fontsize=10)
ax3.set_title("Perbandingan Metrik", fontsize=11)
ax3.legend(fontsize=9)
ax3.set_ylim(0, max(max(nilai_lr), max(nilai_svr)) * 1.25)
ax3.grid(axis="y", linestyle="--", alpha=0.4)
 
# ── 4. Matriks Korelasi ────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
 
kolom_korelasi = ["uhh", "hls", "rls", "pengeluaran_per_kapita", "ipm"]
label_tampil   = ["UHH", "HLS", "RLS", "Pengeluaran", "IPM"]
 
matriks_korel = df[kolom_korelasi].corr()
 
sns.heatmap(
    matriks_korel,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    vmin=0.6, vmax=1.0,
    ax=ax4,
    xticklabels=label_tampil,
    yticklabels=label_tampil,
    linewidths=0.5,
    linecolor="white",
    annot_kws={"size": 9}
)
ax4.set_title("Matriks Korelasi", fontsize=11)
ax4.tick_params(axis="x", rotation=0, labelsize=9)
ax4.tick_params(axis="y", rotation=0, labelsize=9)
 
# ── Simpan & tampilkan ─────────────────────────────────────────────────────
plt.savefig("visualisasi_ipm_sulteng.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nVisualisasi berhasil disimpan ke file: visualisasi_ipm_sulteng.png")