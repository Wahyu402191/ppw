import pandas as pd
import numpy as np
import re
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from collections import Counter

# Fungsi preprocessing teks
def preprocess_text(text):
    """Membersihkan dan preprocessing teks"""
    if pd.isna(text):
        return ""
    # Lowercase
    text = str(text).lower()
    # Hapus karakter khusus, hanya ambil huruf dan spasi
    text = re.sub(r'[^a-z\s]', ' ', text)
    # Hapus spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Baca dataset
print("Membaca dataset...")
df_finance = pd.read_csv('detik_finance_100.csv')
df_sport = pd.read_csv('detik_sport_100.csv')

# Tambahkan label
df_finance['label'] = 1
df_sport['label'] = 2

# Gabungkan dataset
df_combined = pd.concat([df_finance, df_sport], ignore_index=True)
df_combined['ID'] = range(1, len(df_combined) + 1)

print(f"\nTotal dokumen: {len(df_combined)}")
print(f"Dokumen Finance: {len(df_finance)}")
print(f"Dokumen Sport: {len(df_sport)}")

# Preprocessing
print("\nMemproses teks...")
df_combined['teks_bersih'] = df_combined['isi_berita'].apply(preprocess_text)

# Ekstrak semua kata unik dari semua dokumen
all_words = []
for text in df_combined['teks_bersih']:
    words = text.split()
    all_words.extend(words)

# Hitung kata unik
unique_words = list(set(all_words))
total_unique_words = len(unique_words)
print(f"\nJumlah kata unik sebelum filtering: {total_unique_words}")

# Buat matriks Bag of Words (kata unik)
print("\nMembuat matriks Bag of Words...")
word_freq_matrix = []

for idx, text in enumerate(df_combined['teks_bersih']):
    words = text.split()
    word_counts = Counter(words)
    
    row = {
        'ID': idx + 1,
        'dokumen': 'finance' if df_combined.iloc[idx]['label'] == 1 else 'sport'
    }
    
    # Hitung frekuensi setiap kata unik
    for word in unique_words:
        row[word] = word_counts.get(word, 0)
    
    row['label'] = df_combined.iloc[idx]['label']
    word_freq_matrix.append(row)

df_word_freq = pd.DataFrame(word_freq_matrix)

# Simpan tabel Bag of Words
print("Menyimpan tabel Bag of Words...")
df_word_freq.to_csv('output_bow.csv', index=False)

# TF-IDF Transformation
print("\nMenghitung TF-IDF...")

# Filter kata yang terlalu jarang (muncul di < 2 dokumen) atau terlalu sering (> 95% dokumen)
vectorizer = TfidfVectorizer(
    max_features=2000,  # Batasi 2000 fitur
    min_df=2,
    max_df=0.95,
    tokenizer=lambda x: x.split()
)

tfidf_matrix = vectorizer.fit_transform(df_combined['teks_bersih'])
feature_names = vectorizer.get_feature_names_out()

print(f"Jumlah kata unik setelah filtering: {len(feature_names)}")

# Konversi TF-IDF ke DataFrame
tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=feature_names
)

tfidf_df.insert(0, 'ID', range(1, len(tfidf_df) + 1))
tfidf_df.insert(1, 'dokumen', ['finance' if label == 1 else 'sport' for label in df_combined['label']])
tfidf_df['label'] = df_combined['label'].values

# Simpan TF-IDF
print("Menyimpan tabel TF-IDF...")
tfidf_df.to_csv('output_tfidf.csv', index=False)

# PCA - Reduksi Dimensi
print("\nMelakukan PCA (reduksi dimensi)...")
n_components = min(50, len(feature_names), len(df_combined) - 1)  # Maksimal 50 komponen

pca = PCA(n_components=n_components)
tfidf_pca = pca.fit_transform(tfidf_matrix.toarray())

# Buat DataFrame PCA
pca_columns = [f'PC{i+1}' for i in range(n_components)]
pca_df = pd.DataFrame(tfidf_pca, columns=pca_columns)
pca_df.insert(0, 'ID', range(1, len(pca_df) + 1))
pca_df.insert(1, 'dokumen', ['finance' if label == 1 else 'sport' for label in df_combined['label']])
pca_df['label'] = df_combined['label'].values

# Simpan PCA
print("Menyimpan tabel PCA...")
pca_df.to_csv('output_pca.csv', index=False)

# Explained variance
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print(f"\nVariance explained by first 10 components: {cumulative_variance[9]:.2%}")
print(f"Variance explained by all {n_components} components: {cumulative_variance[-1]:.2%}")

# Simpan dataset gabungan dengan preprocessing
df_export = df_combined[['ID', 'kategori', 'judul', 'isi_berita', 'label']].copy()
df_export.to_csv('output_combined_dataset.csv', index=False)

# Buat metadata untuk web
metadata = {
    'total_documents': int(len(df_combined)),
    'finance_documents': int(len(df_finance)),
    'sport_documents': int(len(df_sport)),
    'unique_words_before': int(total_unique_words),
    'unique_words_after': int(len(feature_names)),
    'pca_components': int(n_components),
    'variance_explained_10': float(cumulative_variance[9] if len(cumulative_variance) > 9 else cumulative_variance[-1]),
    'variance_explained_all': float(cumulative_variance[-1])
}

with open('metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("\n" + "="*60)
print("PROSES SELESAI!")
print("="*60)
print(f"Total dokumen: {metadata['total_documents']}")
print(f"Kata unik sebelum filtering: {metadata['unique_words_before']}")
print(f"Kata unik setelah filtering: {metadata['unique_words_after']}")
print(f"Komponen PCA: {metadata['pca_components']}")
print("="*60)
print("\nFile yang dihasilkan:")
print("1. output_combined_dataset.csv - Dataset gabungan")
print("2. output_bow.csv - Bag of Words")
print("3. output_tfidf.csv - TF-IDF Matrix")
print("4. output_pca.csv - PCA Reduced")
print("5. metadata.json - Metadata statistik")
