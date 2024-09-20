# Klasifikasi Gambar Rambu Lalu Lintas🚦

Proyek ini bertujuan untuk membangun model klasifikasi gambar menggunakan Convolutional Neural Networks (CNN) berdasarkan dataset rambu lalu lintas dari Jerman. Model ini dirancang untuk mengidentifikasi jenis-jenis rambu lalu lintas dari gambar dengan presisi tinggi.

## Daftar Isi

- [Pendahuluan](#pendahuluan)
- [Dataset](#dataset)
- [Proses Data](#proses-data)
- [Pemodelan](#pemodelan)
- [Evaluasi](#evaluasi)
- [Kesimpulan](#kesimpulan)

## Pendahuluan

Studi ini bertujuan untuk menerapkan teknik deep learning pada tugas klasifikasi gambar rambu lalu lintas. Dataset berisi lebih dari 50.000 gambar yang terbagi dalam 43 kelas, seperti batas kecepatan, bahaya, dan tanda berhenti. Setiap gambar diambil dari foto rambu jalanan di dunia nyata, bukan sintetis.

## Dataset

Dataset yang digunakan dalam proyek ini berasal dari German Traffic Sign Dataset (Official). Dataset ini mencakup ribuan gambar rambu lalu lintas dengan variasi bentuk, warna, dan kategori rambu.

Untuk informasi lebih lanjut tentang dataset, kunjungi [German Traffic Sign Dataset](https://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) atau [Kaggle](https://www.kaggle.com/datasets/saadhaxxan/germantrafficsigns).

### Distribusi Data

Dataset terdiri dari gambar-gambar dalam resolusi rendah yang telah diurutkan berdasarkan jenis rambu. Dataset ini terbagi menjadi:

- Training set: digunakan untuk melatih model.
- Validation set: digunakan untuk menguji kinerja model selama proses pelatihan.
- Test set: digunakan untuk mengevaluasi kinerja model setelah pelatihan.

Berikut adalah distribusi data di setiap subset (training, validasi, testing):

| Subset     | Jumlah Gambar |
| ---------- | ------------- |
| Training   | 90%           |
| Validation | 10%           |
| Testing    | 100%          |

## Proses Data

### Loading Dataset

Dataset dimuat dari file dengan format .p dan .csv. Label gambar digunakan untuk mengklasifikasikan gambar ke dalam kelas yang sesuai. Berikut adalah langkah-langkah utama:

1. Ekstraksi Data: Dataset diload ke dalam format numpy array untuk kemudian diolah.
2. Pembagian Data: Dataset dibagi menjadi set pelatihan, validasi, dan testing menggunakan train_test_split.
3. One-Hot Encoding: Label klasifikasi dikonversi menjadi format one-hot encoding agar bisa digunakan dalam CNN.

### Pra-pemrosesan Data

Beberapa langkah pra-pemrosesan data yang dilakukan meliputi:

- Normalisasi Data: Data gambar yang terdiri dari piksel dinormalisasi agar berada dalam rentang 0 hingga 1.
- Augmentasi Gambar: Untuk meningkatkan kinerja model, augmentasi gambar dapat digunakan, namun dalam proyek ini augmentasi tidak diterapkan.

## Pemodelan

Model CNN dibangun menggunakan Keras dengan arsitektur sebagai berikut:

- Conv2D: Layer convolutional untuk mendeteksi fitur dari gambar.
- MaxPooling2D: Layer pooling untuk mengurangi dimensi.
- Dropout: Teknik regularisasi untuk menghindari overfitting.
- Dense (Fully Connected Layer): Menerapkan fungsi aktivasi ReLU dan softmax untuk output multi-kelas.

### Arsitektur Model

- Input layer: Gambar berukuran 32x32x3 (RGB).
- Convolutional layers: 4 lapisan convolusi dengan kernel masing-masing 32 dan 64 filter.
- Pooling layers: 2 lapisan pooling untuk mengurangi ukuran feature map.
- Fully connected layers: Menghubungkan output convolutional layers ke layer dense untuk klasifikasi akhir.
- Output layer: Softmax activation untuk prediksi 43 kelas.

```python
model = Sequential([
    Conv2D(32, (5, 5), activation='relu', input_shape=X_train.shape[1:]),
    Conv2D(32, (5, 5), activation='relu'),
    MaxPool2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu'),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPool2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(43, activation='softmax')
])
```

### Pelatihan Model

Model dilatih menggunakan data training dan divalidasi dengan validation set. Proses pelatihan berhenti ketika akurasi mencapai >96%.

```python
history = model.fit(X_train, y_train, batch_size=32, epochs=25, validation_data=(X_val, y_val), callbacks=[callbacks])
```

<img src="https://github.com/user-attachments/assets/8c6c1956-243b-4791-a309-d0b6f120b7f0" alt="acc" width="25%"/>
<img src="https://github.com/user-attachments/assets/6a210195-e5f7-4cdf-a261-583f6f681642" alt="loss" width="25%"/>

## Evaluasi

Model diuji pada data testing untuk mengukur akurasi prediksi. Metrik evaluasi utama adalah accuracy dan classification report yang meliputi precision, recall, dan F1-score.

```python
accuracy_score(y_test, pred)
```

## Kesimpulan

Model CNN berhasil mencapai akurasi lebih dari 96% pada data validasi. Untuk pengembangan lebih lanjut, disarankan untuk melakukan augmentasi data dan hyperparameter tuning untuk meningkatkan kinerja model.

Dengan hasil ini, model mampu mengklasifikasikan gambar rambu lalu lintas dengan akurasi tinggi, yang dapat berguna dalam pengembangan sistem pengenalan rambu untuk kendaraan otonom.
