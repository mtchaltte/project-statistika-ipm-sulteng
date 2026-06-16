Prediksi IPM Sulawesi Tengah

Proyek ini dibuat untuk memprediksi nilai Indeks Pembangunan Manusia (IPM) di kabupaten/kota Sulawesi Tengah menggunakan data BPS tahun 2019–2023.

Saya menggunakan dua model utama:


Regresi Linear untuk memprediksi nilai IPM secara numerik
Support Vector Classification (SVC) untuk memprediksi kategori IPM (Sedang / Tinggi / Sangat Tinggi)


Selain itu saya juga menambahkan SVR (Support Vector Regression) sebagai pembanding Regresi Linear melalui grafik visualisasi.


Struktur File

├── prediksi_ipm_sulteng.py                     # script utama
├── Dataset_IPM_Sulawesi_Tengah_2019-2023.xlsx  # dataset dari BPS
├── hasil_prediksi_ipm_2026.xlsx                # hasil prediksi tahun 2026
└── visualisasi_ipm_sulteng.png                 # grafik visualisasi


Cara Menjalankan

Install library yang dibutuhkan terlebih dahulu:

bashpip install pandas numpy matplotlib seaborn scikit-learn openpyxl

Jika dijalankan di lokal, sesuaikan path dataset di bagian awal script:

pythonfile_path = r"D:\Tugas\Dataset_IPM_Sulawesi_Tengah_2019-2023.xlsx"  # contoh Windows

Jika menggunakan Google Colab, upload file Excel-nya terlebih dahulu lalu ubah path-nya:

pythonfile_path = "/content/Dataset_IPM_Sulawesi_Tengah_2019-2023.xlsx"

Setelah itu jalankan scriptnya, dua file output akan otomatis tersimpan di folder yang sama.


Penjelasan Kode

Membaca dan Membersihkan Data

Dataset dibaca dari sheet "Data Utama" dengan header di baris ke-3. Nama kolom aslinya mengandung karakter enter dan spasi berlebih, sehingga saya rapikan menjadi nama yang lebih sederhana seperti uhh, hls, rls, dan sebagainya. Kolom no juga dihapus karena tidak digunakan.

Fitur yang Digunakan

FiturKeterangantahunTahun data (2019–2023)kabupaten_kotaNama kabupaten/kotauhhUmur Harapan Hidup (tahun)hlsHarapan Lama Sekolah (tahun)rlsRata-rata Lama Sekolah (tahun)pengeluaran_per_kapitaPengeluaran per kapita (ribu Rp/tahun)

Untuk preprocessing, fitur numerik dinormalisasi menggunakan StandardScaler, sedangkan kolom kabupaten_kota dikonversi ke bentuk one-hot encoding karena berisi data teks.

Model Regresi Linear

Digunakan untuk memprediksi nilai IPM secara langsung. Data dibagi dengan rasio 80:20 untuk pelatihan dan pengujian. Evaluasi menggunakan MAE, RMSE, dan R².

Model SVC

Digunakan untuk memprediksi kategori IPM. Kolom kategori_ipm dalam dataset sudah tersedia dan sesuai standar BPS sehingga bisa langsung dipakai. Saya menggunakan stratify saat membagi data agar proporsi tiap kategori tetap seimbang di data latih maupun uji.

Model SVR

Ditambahkan khusus untuk keperluan visualisasi, agar hasil prediksinya bisa dibandingkan dengan Regresi Linear melalui scatter plot.

Prediksi Tahun 2026

Nilai fitur untuk tahun 2026 diestimasi menggunakan rata-rata dari tiga tahun terakhir (2021–2023) per kabupaten/kota, kemudian dimasukkan ke model untuk menghasilkan prediksi IPM dan kategorinya.


Visualisasi

Script akan menghasilkan satu gambar berisi empat grafik yang disimpan sebagai visualisasi_ipm_sulteng.png:


Scatter plot Regresi Linear — perbandingan IPM aktual vs prediksi dengan garis ideal
Scatter plot SVR — sama seperti di atas menggunakan model SVR
Bar chart perbandingan metrik — MAE, RMSE, dan R² kedua model secara berdampingan
Matriks korelasi — heatmap hubungan antar fitur numerik dengan IPM



Output

Dua file yang dihasilkan setelah script selesai dijalankan:


hasil_prediksi_ipm_2026.xlsx — tabel prediksi nilai IPM dan kategori per kabupaten/kota
visualisasi_ipm_sulteng.png — grafik perbandingan model dan analisis korelasi
