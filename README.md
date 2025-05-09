## Struktur Direktori
```bash
├── train.py
├── custom_trainer.py
├── utils.py
├── predict.py
├── evaluate_model.py
├── dataset/
│ ├── data-kemasan.json
│ ├── kemasan_1.jpg
│ ├── dst...
│ ├── train_data.json
│ └── test_data.json
└── requirements.txt
```

## deskripsi
### 1. `train.py`
Script utama untuk melakukan pelatihan (training) model TrOCR dengan dataset kemasan. File ini berisi:
- **Data loading**: Mengambil data gambar dan teks dari file `data-kemasan.json`.
- **Preprocessing**: Mengonversi gambar ke format tensor dan mempersiapkan label.
- **Trainer Setup**: Mengonfigurasi `Seq2SeqTrainer` untuk melatih model.
- **Model Training**: Melatih model menggunakan dataset train dan evaluasi menggunakan dataset test.

### 2. `custom_trainer.py`
Kelas `CustomTrainer` yang meng-override metode `compute_loss` dari `Seq2SeqTrainer`. Fungsi ini bertujuan untuk menyesuaikan perhitungan loss selama pelatihan dengan memperhatikan input dan output model yang berbeda-beda.

### 3. `utils.py`
File ini berisi fungsi-fungsi bantuan:
- **`clean_text`**: Fungsi untuk membersihkan teks dari karakter yang tidak diinginkan, seperti simbol dan angka.
- **`preprocess`**: Fungsi untuk mempersiapkan gambar dan label, mengubah gambar menjadi tensor dan label menjadi ID token untuk model.

### 4. `predict.py`
Script untuk melakukan prediksi menggunakan model yang telah dilatih. File ini memuat:
- Pemrosesan gambar input.
- Penggunaan model yang telah dilatih untuk menghasilkan prediksi teks.
- Menampilkan hasil prediksi.

### 5. `evaluate_model.py`
File ini digunakan untuk mengevaluasi hasil model dengan menghitung metrik seperti **Character Error Rate (CER)** dan **Word Error Rate (WER)**. Script ini juga mencetak hasil prediksi dan perbandingannya dengan teks asli (referensi).

### 6. `dataset/`
Folder ini berisi dataset yang digunakan untuk melatih dan menguji model:
- **`data-kemasan.json`**: File JSON yang berisi daftar gambar dan teks untuk pelatihan.
- **`train_data.json`**: Dataset pelatihan setelah diproses.
- **`test_data.json`**: Dataset pengujian setelah diproses.
