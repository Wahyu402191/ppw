# Text Mining Analysis - Web Mining Project

Repository ini berisi analisis lengkap Text Mining pada dataset berita Finance dan Sport menggunakan teknik preprocessing, TF-IDF, dan PCA.

## Profile

- **Nama:** Wahyu Pratama
- **NPM:** 230411100058
- **Email:** wahyupritim@gmail.com
- **Mata Kuliah:** Web Mining

## Deskripsi Proyek

Proyek ini melakukan analisis text mining pada 200 dokumen berita yang terdiri dari:
- 100 berita kategori **Finance** (Label 1)
- 100 berita kategori **Sport** (Label 2)

### Proses Analisis

1. **Preprocessing**
   - Cleaning text (lowercase, remove special characters)
   - Tokenization
   - Ekstraksi kata unik

2. **Bag of Words (BoW)**
   - Menghitung frekuensi kemunculan setiap kata unik pada setiap dokumen
   - Menghasilkan matriks dengan dimensi: 200 dokumen × kata unik

3. **TF-IDF (Term Frequency - Inverse Document Frequency)**
   - Representasi dokumen menggunakan TF-IDF
   - Filtering kata yang terlalu jarang (muncul di < 2 dokumen) atau terlalu umum (> 95% dokumen)
   - Mengurangi dimensi dari 7,321 kata menjadi 2,000 fitur terbaik

4. **PCA (Principal Component Analysis)**
   - Reduksi dimensi dari 2,000 fitur menjadi 50 komponen utama
   - Mempertahankan 52.63% variance dari data original
   - 10 komponen pertama menjelaskan 19.55% variance

## Statistik Dataset

| Metrik | Nilai |
|--------|-------|
| Total Dokumen | 200 |
| Dokumen Finance | 100 |
| Dokumen Sport | 100 |
| Kata Unik (Sebelum Filtering) | 7,321 |
| Kata Unik (Setelah Filtering) | 2,000 |
| Komponen PCA | 50 |
| Variance Explained (10 PC) | 19.55% |
| Variance Explained (50 PC) | 52.63% |

## File Output

### Dataset
- `detik_finance_100.csv` - Dataset original berita finance
- `detik_sport_100.csv` - Dataset original berita sport
- `output_combined_dataset.csv` - Dataset gabungan dengan label

### Hasil Analisis
- `output_bow.csv` - Bag of Words (7,321 kolom kata unik)
- `output_tfidf.csv` - TF-IDF Matrix (2,000 fitur)
- `output_pca.csv` - PCA Reduced (50 komponen)
- `metadata.json` - Statistik dan metadata

## Struktur File

```
ppw/
├── index.html                      # Website visualisasi
├── process_data.py                 # Script preprocessing & analisis
├── detik_finance_100.csv          # Dataset finance
├── detik_sport_100.csv            # Dataset sport
├── output_combined_dataset.csv    # Dataset gabungan
├── output_bow.csv                 # Bag of Words
├── output_tfidf.csv               # TF-IDF Matrix
├── output_pca.csv                 # PCA Result
├── metadata.json                  # Statistik
└── README.md                      # Dokumentasi
```

## Cara Menjalankan

### 1. Preprocessing Data

```bash
python process_data.py
```

Script ini akan:
- Membaca dataset finance dan sport
- Melakukan preprocessing (cleaning, tokenization)
- Membuat Bag of Words
- Menghitung TF-IDF
- Melakukan PCA
- Menyimpan semua hasil ke file CSV

### 2. Visualisasi Web

Buka file `index.html` di browser untuk melihat:
- Profile mahasiswa
- Materi Web Mining
- Dataset finance dan sport
- Tabel Bag of Words
- Tabel TF-IDF
- Tabel PCA

## Teknologi yang Digunakan

### Python Libraries
- **pandas** - Data manipulation
- **numpy** - Numerical computation
- **scikit-learn** - TF-IDF dan PCA
- **re** - Regular expressions untuk text cleaning

### Web Technologies
- **HTML5** - Structure
- **CSS3** - Styling dengan gradient modern
- **JavaScript** - Dynamic table loading

## Penjelasan Teknik

### TF-IDF (Term Frequency - Inverse Document Frequency)

TF-IDF adalah metode statistik untuk mengevaluasi seberapa penting sebuah kata terhadap dokumen dalam collection.

**Formula:**
```
TF-IDF(t,d) = TF(t,d) × IDF(t)

TF(t,d) = (Jumlah kemunculan kata t di dokumen d) / (Total kata di dokumen d)
IDF(t) = log(Total dokumen / Jumlah dokumen yang mengandung kata t)
```

### PCA (Principal Component Analysis)

PCA adalah teknik reduksi dimensi yang mentransformasi data ke dalam sistem koordinat baru dimana:
- Komponen pertama memiliki variance terbesar
- Setiap komponen berikutnya memiliki variance terbesar berikutnya dan orthogonal dengan komponen sebelumnya

**Keuntungan:**
- Mengurangi dimensi data (dari 2,000 ke 50)
- Menghilangkan noise
- Mempercepat komputasi untuk model machine learning
- Memvisualisasikan data high-dimensional

## Web Mining

### Apa itu Web Mining?

Web Mining merupakan proses penerapan teknik data mining untuk menemukan informasi, pola, dan pengetahuan yang bermanfaat dari data yang terdapat di World Wide Web.

### Tiga Jenis Web Mining

1. **Web Content Mining**
   - Proses menemukan informasi dari isi atau konten halaman web
   - Analisis teks, gambar, video, dan multimedia lainnya

2. **Web Structure Mining**
   - Menganalisis struktur dan hubungan antarhalaman web
   - Analisis hyperlink dan struktur website

3. **Web Usage Mining**
   - Menganalisis aktivitas dan pola perilaku pengguna
   - Analisis log server dan clickstream data

## Aplikasi Text Mining

Text mining pada proyek ini dapat diaplikasikan untuk:
- Klasifikasi dokumen otomatis (Finance vs Sport)
- Clustering berita berdasarkan topik
- Information retrieval dan search engine
- Analisis sentiment
- Topic modeling
- Rekomendasi konten

## Kesimpulan

Proyek ini mendemonstrasikan pipeline lengkap text mining:
1. Mengumpulkan dan menggabungkan dataset
2. Preprocessing dan cleaning text
3. Feature extraction dengan Bag of Words dan TF-IDF
4. Dimensionality reduction dengan PCA
5. Visualisasi hasil dalam web interface yang elegan

Hasil analisis menunjukkan bahwa dengan 2,000 fitur TF-IDF yang direduksi menjadi 50 komponen PCA, kita dapat mempertahankan 52.63% informasi dari data original, yang cukup untuk berbagai aplikasi machine learning.

## Lisensi

Project ini dibuat untuk keperluan akademik - Tugas Mata Kuliah Web Mining 2026

---

© 2026 Wahyu Pratama • NPM 230411100058
