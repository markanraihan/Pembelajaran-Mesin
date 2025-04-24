# Decision‑Tree Classifier: Citrus vs Grapes

Model machine learning sederhana untuk mengklasifikasikan buah menjadi jeruk atau anggur menggunakan Decision Tree Classifier dari scikit‑learn.

---

## Daftar Isi
1. Ringkasan Proyek  
2. Dataset  
3. Instalasi  
4. Cara Menjalankan  
5. Pipeline Pemodelan  
6. Hasil & Contoh Output  
7. Struktur Proyek  
8. Bio

---

## 1. Ringkasan Proyek
Repositori ini berisi kode, data, dan dokumentasi untuk membangun model klasifikasi buah (jeruk vs anggur). Model dilatih dengan Decision Tree dilengkapi GridSearchCV guna mencari hiper‑parameter terbaik.

---

## 2. Dataset
File data: `citrus.csv`

Kolom penting  
- `label` : target (jeruk → 0, anggur → 1)  
- fitur lain : tinggi, lebar, warna, berat, dll (silakan lihat header CSV)

---

## 3. Instalasi

```bash
git clone https://github.com/username/repo-citrus.git
cd repo-citrus
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 4. Cara Menjalankan
Latih model dari nol:

```bash
python decision_tree_citrus.py --data citrus.csv
```

Argumen penting  
- `--data` : path ke file CSV (default =citrus.csv)  
- `--save` : nama file output model (default =model.joblib)  

---

## 5. Pipeline Pemodelan
1. Import pustaka & dataset  
2. Pra‑proses  
   - menangani nilai hilang  
   - encoding label (jeruk = 0, anggur = 1)  
3. Train‑test split (80 / 20, stratified)  
4. Pemodelan  
   - `DecisionTreeClassifier`  
   - pencarian grid:  
     - `criterion` : gini, entropy  
     - `max_depth` : 3, 5, 7, None  
     - `min_samples_split` : 2, 5, 10  
     - `min_samples_leaf` : 1, 2, 4  
5. Evaluasi  
   - accuracy, precision, recall, f1  
   - confusion matrix  
   - feature importance plot  
6. Simpan model (`joblib.dump`)

---

## 6. Hasil & Contoh Output
Setelah pelatihan, skrip menampilkan metrik di terminal serta menyimpan:
- `model.joblib` : model terlatih  
- `confusion_matrix.png`  
- `feature_importance.png`  

Contoh akurasi: 0.93 (akan bervariasi tergantung data).

---

## 7. Struktur Proyek
```
repo-citrus/
├── citrus.csv
├── decision_tree_citrus.py
├── requirements.txt
├── README.md
└── outputs/
    ├── model.joblib
    ├── confusion_matrix.png
    └── feature_importance.png
```

---

## 8. Bio
UTS Mata Kuliah Praktikum Pembelajaran Mesin

Nama: Muhammad Arkan Raihan

NIM: 1227050085

Kelas: Praktikum Pembelajaran Mesin E

## 9. Kesimpulan
Pada praktik ini saya membangun model *Decision Tree* untuk klasifikasi buah sitrus (jeruk vs grapefruit). 
Langkah‑langkahnya:
a. Load data & label encoding
   - Dataset `citrus.csv` dibaca.  
   - Kolom `name` dijadikan label; nilai uniknya `orange` & `grapefruit`, lalu di‑encode.

b. Split data
   - Data dipisah 80 % train : 20 % test pakai `train_test_split` dengan stratifikasi.

c. Tuning hyper‑parameter
   - GridSearchCV (5‑fold) mencari kombinasi terbaik.  
   - Setting optimal: `criterion=entropy`, `max_depth=None`, `min_samples_split=5`, `min_samples_leaf=1`.

d. Evaluasi  
   - Akurasi pada data uji: **96 %**.  
   - Precision, recall, f1‑score untuk kedua kelas ~0.95–0.96 → model tergolong sangat baik.

e. *Output 
   - Confusion matrix & feature‑importance disimpan sebagai PNG.  
   - Model final disimpan ke `model_dt.joblib` untuk deployment selanjutnya.

Intinya, model *Decision Tree* yang dihasilkan sudah mampu membedakan jeruk dan grapefruit dengan performa tinggi (96 % akurasi) dan siap dipakai / diekspor ke produksi.
