# Tutorial: Cara Menjalankan Text Mining Analysis

## Prasyarat

Sebelum memulai, pastikan Anda sudah menginstall:

1. **Python 3.x** (disarankan Python 3.7 atau lebih baru)
2. **Library Python yang diperlukan:**
   ```bash
   pip install pandas numpy scikit-learn
   ```

## Langkah-Langkah

### 1. Clone Repository

```bash
git clone https://github.com/Wahyu402191/ppw.git
cd ppw
```

### 2. Jalankan Preprocessing Data

Script `process_data.py` akan memproses dataset dan menghasilkan semua file analisis yang diperlukan.

```bash
python process_data.py
```

**Output yang dihasilkan:**
- `output_combined_dataset.csv` - Dataset gabungan (200 dokumen)
- `output_bow.csv` - Bag of Words dengan 7,321 kata unik
- `output_tfidf.csv` - TF-IDF Matrix dengan 2,000 fitur terfilter
- `output_pca.csv` - PCA dengan 50 komponen
- `metadata.json` - Statistik analisis

**Proses yang dilakukan:**
1. Membaca dataset finance (100 dokumen) dan sport (100 dokumen)
2. Preprocessing teks (lowercase, remove special characters)
3. Ekstraksi kata unik (7,321 kata ditemukan)
4. Membuat Bag of Words matrix
5. Menghitung TF-IDF dan filtering (2,000 fitur terbaik)
6. Reduksi dimensi dengan PCA (50 komponen)

### 3. Buka Website

Setelah preprocessing selesai, buka file `index.html` di browser Anda.

**Cara 1: Double-click**
- Cukup double-click file `index.html`

**Cara 2: Menggunakan Python HTTP Server**
```bash
python -m http.server 8000
```
Kemudian buka browser dan akses: `http://localhost:8000`

**Cara 3: Menggunakan Live Server (VS Code)**
- Install extension "Live Server" di VS Code
- Klik kanan pada `index.html`
- Pilih "Open with Live Server"

### 4. Navigasi Website

Website terdiri dari 5 section utama:

#### a. **Home (Hero Section)**
- Pengantar project
- Navigasi cepat ke bagian lain

#### b. **Profile**
- Informasi mahasiswa
- NPM, Email, Mata Kuliah

#### c. **Materi**
- Penjelasan Web Mining
- 3 Jenis Web Mining:
  - Web Content Mining
  - Web Structure Mining  
  - Web Usage Mining

#### d. **Dataset**
- Statistik dataset (200 dokumen, 7,321 → 2,000 kata)
- Tabel dataset Finance (100 dokumen dengan Label 1)
- Tabel dataset Sport (100 dokumen dengan Label 2)
- Fitur: scroll horizontal untuk kolom yang panjang

#### e. **Analysis**
- Tabel Bag of Words (frekuensi kata per dokumen)
- Tabel TF-IDF (nilai TF-IDF untuk setiap kata)
- Tabel PCA (50 komponen utama)
- Semua tabel dapat discroll horizontal dan vertical

## Penjelasan File Output

### 1. output_combined_dataset.csv
Dataset gabungan dari finance + sport dengan struktur:
```
ID, kategori, judul, isi_berita, label
1, Infrastruktur, ..., ..., 1
...
100, ..., ..., ..., 1
101, Raket, ..., ..., 2
...
200, ..., ..., ..., 2
```

### 2. output_bow.csv
Bag of Words - frekuensi kemunculan kata:
```
ID, dokumen, kata1, kata2, ..., kata7321, label
1, finance, 5, 0, ..., 2, 1
2, finance, 3, 1, ..., 0, 1
...
```
- **Kolom:** ID, dokumen, 7,321 kata unik, label
- **Nilai:** Jumlah kemunculan kata di setiap dokumen

### 3. output_tfidf.csv
TF-IDF Matrix:
```
ID, dokumen, kata1, kata2, ..., kata2000, label
1, finance, 0.123, 0.0, ..., 0.456, 1
2, finance, 0.089, 0.234, ..., 0.0, 1
...
```
- **Kolom:** ID, dokumen, 2,000 fitur terpilih, label
- **Nilai:** TF-IDF score (0.0 - 1.0)
- **Formula:** TF-IDF = TF × log(N/df)
  - TF = Term Frequency (frekuensi kata di dokumen)
  - N = Total dokumen
  - df = Document Frequency (jumlah dokumen yang mengandung kata)

### 4. output_pca.csv
Principal Component Analysis:
```
ID, dokumen, PC1, PC2, ..., PC50, label
1, finance, 0.123, -0.456, ..., 0.789, 1
2, finance, -0.234, 0.567, ..., -0.123, 1
...
```
- **Kolom:** ID, dokumen, 50 komponen utama, label
- **Nilai:** Skor komponen principal (-∞ to +∞)
- **Variance Explained:** 52.63% dari data original

### 5. metadata.json
Statistik analisis:
```json
{
  "total_documents": 200,
  "finance_documents": 100,
  "sport_documents": 100,
  "unique_words_before": 7321,
  "unique_words_after": 2000,
  "pca_components": 50,
  "variance_explained_10": 0.1955,
  "variance_explained_all": 0.5263
}
```

## Memahami Proses Text Mining

### 1. Preprocessing
Membersihkan teks agar siap diproses:
- **Lowercase:** "Bank" → "bank"
- **Remove special chars:** "harga: Rp10.000" → "harga rp"
- **Tokenization:** "bank indonesia" → ["bank", "indonesia"]

### 2. Bag of Words (BoW)
Merepresentasikan dokumen sebagai frekuensi kata:

**Contoh:**
```
Dokumen 1: "bank indonesia menetapkan suku bunga"
Dokumen 2: "suku bunga bank naik"

BoW Matrix:
         bank  indonesia  menetapkan  suku  bunga  naik
Dok 1     1       1          1         1     1      0
Dok 2     1       0          0         1     1      1
```

### 3. TF-IDF
Memberikan bobot lebih pada kata yang penting:
- Kata yang sering muncul di 1 dokumen tapi jarang di dokumen lain → bobot tinggi
- Kata yang muncul di semua dokumen (seperti "dan", "yang") → bobot rendah

**Contoh:**
```
Kata "indonesia" muncul 10x di dokumen 1, tapi hanya di 5 dari 200 dokumen
→ TF-IDF tinggi (kata penting untuk dokumen ini)

Kata "dan" muncul 20x di dokumen 1, dan ada di 190 dari 200 dokumen
→ TF-IDF rendah (kata terlalu umum)
```

### 4. PCA (Dimensionality Reduction)
Mengurangi jumlah fitur sambil mempertahankan informasi:
- **Sebelum:** 2,000 fitur (kata)
- **Sesudah:** 50 komponen
- **Information retained:** 52.63%

**Keuntungan:**
- Mengurangi noise
- Mempercepat training model ML
- Menghilangkan redundansi (kata yang berkorelasi tinggi)

## Troubleshooting

### Error: ModuleNotFoundError
```
ModuleNotFoundError: No module named 'sklearn'
```
**Solusi:**
```bash
pip install scikit-learn
```

### Error: File not found
```
FileNotFoundError: detik_finance_100.csv
```
**Solusi:**
- Pastikan Anda menjalankan script dari direktori `ppw`
- Check apakah file CSV ada di folder yang sama

### Website tidak memuat data
**Solusi:**
1. Pastikan file output_*.csv sudah dihasilkan
2. Buka browser console (F12) untuk melihat error
3. Coba gunakan Python HTTP server daripada double-click file

### Data tidak muncul di tabel
**Solusi:**
1. Tunggu beberapa detik (file CSV besar memerlukan waktu loading)
2. Refresh browser (Ctrl+F5)
3. Check console browser untuk error JavaScript

## Tips & Tricks

### 1. Membatasi Rows yang Ditampilkan
Edit `index.html`, cari fungsi `createTable` dan ubah parameter `maxRows`:
```javascript
// Tampilkan hanya 20 baris pertama
document.getElementById('bow-table').innerHTML = createTable(limitedHeaders, bowData.data, 20);
```

### 2. Mengubah Jumlah Komponen PCA
Edit `process_data.py`, cari baris:
```python
n_components = min(50, len(feature_names), len(df_combined) - 1)
```
Ubah 50 menjadi jumlah yang diinginkan (misal 100)

### 3. Mengubah Filtering TF-IDF
Edit `process_data.py`, cari:
```python
vectorizer = TfidfVectorizer(
    max_features=2000,  # Ubah ini
    min_df=2,           # Minimum dokumen yang harus mengandung kata
    max_df=0.95,        # Maksimum 95% dokumen boleh mengandung kata
    tokenizer=lambda x: x.split()
)
```

### 4. Export ke Excel
Tambahkan di akhir `process_data.py`:
```python
# Export ke Excel
with pd.ExcelWriter('output_all.xlsx') as writer:
    df_word_freq.to_excel(writer, sheet_name='BoW', index=False)
    tfidf_df.to_excel(writer, sheet_name='TF-IDF', index=False)
    pca_df.to_excel(writer, sheet_name='PCA', index=False)
```

## Resources

### Dokumentasi Library
- [Pandas](https://pandas.pydata.org/docs/)
- [Scikit-learn TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [Scikit-learn PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)

### Tutorial Text Mining
- [Introduction to Text Mining](https://towardsdatascience.com/introduction-to-text-mining-5d5f7f7c7f7d)
- [TF-IDF Explained](https://towardsdatascience.com/tf-idf-for-document-ranking-from-scratch-in-python-on-real-world-dataset-796d339a4089)
- [PCA Step-by-Step](https://builtin.com/data-science/step-step-explanation-principal-component-analysis)

## Kontak

Jika ada pertanyaan atau masalah:
- **Email:** wahyupritim@gmail.com
- **GitHub:** [https://github.com/Wahyu402191](https://github.com/Wahyu402191)

---

© 2026 Wahyu Pratama • NPM 230411100058
