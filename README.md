<img width="954" height="622" alt="Screenshot 2026-09-28 at 19 11 59" src="https://github.com/user-attachments/assets/f24798be-158b-4079-a264-718d59c499e8" /># car-plane-conditional-gan
Generasi citra mobil dan pesawat menggunakan Conditional GAN, mencakup baseline, dua modifikasi arsitektur, hyperparameter tuning, serta evaluasi FID.

# Car & Plane Image Generation with Conditional GAN

Proyek akademik yang mengeksplorasi **Conditional Generative Adversarial Network (CGAN)** untuk menghasilkan citra grayscale mobil dan pesawat berdasarkan label kelas. Proyek mencakup exploratory data analysis, implementasi baseline, dua modifikasi model, serta evaluasi menggunakan **Fréchet Inception Distance (FID)** dan visualisasi hasil generasi.

## Gambaran Proyek

CGAN terdiri dari dua jaringan yang dilatih secara adversarial:

- **Generator** menerima random noise dan label kelas, kemudian menghasilkan gambar baru.
- **Discriminator** menerima gambar beserta label kelas dan memprediksi apakah gambar tersebut berasal dari dataset atau generator.

Label `0` merepresentasikan **car**, sedangkan label `1` merepresentasikan **plane**. Label berfungsi sebagai kondisi untuk proses generasi. Proyek ini menggunakan jaringan fully connected, bukan convolutional GAN.

## Dataset dan Preprocessing

Dataset **Car Plane version 2** disediakan dalam tugas akademik. Sumber publik asli dataset belum tercantum dalam notebook.

| Kelas | Training | Test | Total |
|---|---:|---:|---:|
| Car | 7.116 | 896 | 8.012 |
| Plane | 7.124 | 888 | 8.012 |
| **Total** | **14.240** | **1.784** | **16.024** |

Pembagian mengikuti folder `Train` dan `Test` yang tersedia; notebook tidak membuat validation set terpisah.

Tahapan persiapan data:

1. Memeriksa distribusi kelas, mode warna, ukuran gambar, dan file rusak. Pemeriksaan file tidak menemukan gambar rusak pada kedua folder.
2. Memvisualisasikan contoh gambar, distribusi intensitas piksel, dan rata-rata gambar per kelas.
3. Memuat gambar grayscale berukuran **28 × 28 × 1** menggunakan TensorFlow.
4. Menormalisasi piksel dari 0–255 menjadi **−1 hingga 1**, sesuai aktivasi `tanh` pada output generator.
5. Membentuk batch berukuran **64**; training diacak, sedangkan test tidak diacak.

<img width="1026" height="461" alt="Screenshot 2026-09-28 at 19 09 59" src="https://github.com/user-attachments/assets/636e1452-2c7b-42c0-b932-24be27b2eb61" />

## Baseline CGAN

### Generator

```text
Random noise (100 dimensi) + label embedding (2 dimensi)
→ Concatenate
→ Dense 128 → LeakyReLU
→ Dense 256 → LeakyReLU
→ Dense 512 → LeakyReLU
→ Dense 1024 → LeakyReLU
→ Dense 784 → Tanh
→ Reshape (28 × 28 × 1)
```

### Discriminator

```text
Image (28 × 28 × 1) → Flatten (784)
+ label embedding (2 dimensi)
→ Concatenate
→ Dense 512 → LeakyReLU
→ Dense 1024 → LeakyReLU
→ Dense 1024 → LeakyReLU
→ Dense 512 → LeakyReLU
→ Dense 1 → Sigmoid
```

Kedua jaringan menggunakan LeakyReLU dengan negative slope **0,2**, optimizer **Adam** dengan learning rate **0,0002** dan `beta_1=0.5`. Baseline dilatih selama **40 epoch**.

Training menggunakan custom loop dengan `tf.GradientTape` dan **Binary Cross-Entropy**. Generator diarahkan agar gambar buatannya dianggap real, sedangkan discriminator dilatih membedakan pasangan gambar-label real dan generated. Discriminator loss merupakan penjumlahan loss real dan fake.

<img width="808" height="502" alt="Screenshot 2026-09-28 at 19 10 25" src="https://github.com/user-attachments/assets/20f7a1a0-6884-4532-99af-e7804ab1dd95" />
<img width="808" height="502" alt="Screenshot 2026-09-28 at 19 10 43" src="https://github.com/user-attachments/assets/8983c01a-a579-4aa6-9174-88fa9303dcb5" />



<img width="696" height="464" alt="Screenshot 2026-09-28 at 19 11 28" src="https://github.com/user-attachments/assets/bb8dcc0f-e3f5-4614-9972-8ff0985440c5" />

## Modifikasi 1 — Modified CGAN V1

V1 mempertahankan urutan ukuran hidden layer baseline, lalu mengubah regularisasi dan pengaturan training.

| Komponen | Baseline | Modified V1 | Tujuan perubahan |
|---|---|---|---|
| Noise dimension | 100 | 128 | Menambah dimensi input acak yang dapat digunakan generator untuk merepresentasikan variasi gambar |
| Generator normalization | Tidak ada | Batch Normalization setelah setiap hidden Dense layer, sebelum LeakyReLU | Membantu mengendalikan distribusi aktivasi selama training |
| Discriminator regularization | Tidak ada | Dropout 0,3 setelah dua hidden layer pertama | Mengurangi ketergantungan pada aktivasi tertentu dan mencoba membatasi dominasi discriminator |
| Learning rate G dan D | 0,0002 | 0,0001 | Memperkecil langkah pembaruan bobot |
| Training duration | 40 epoch | 70 epoch | Memberi model lebih banyak iterasi pembelajaran |

Label embedding tetap **2 dimensi**. Hidden layer generator tetap **128 → 256 → 512 → 1024**, dan discriminator tetap **512 → 1024 → 1024 → 512**. Batch size tetap 64 dan Adam `beta_1` tetap 0,5.

### Hasil V1

FID yang tercatat turun dari **302,88 menjadi 265,36**, setara penurunan relatif sekitar **12,39%**. Ini menunjukkan jarak distribusi fitur yang lebih kecil pada sampel evaluasi yang digunakan.

Tujuan setiap perubahan di atas adalah alasan desain, bukan bukti bahwa satu perubahan tertentu menyebabkan peningkatan. Beberapa komponen serta jumlah epoch diubah sekaligus, sehingga kontribusinya belum dipisahkan melalui ablation study.

<img width="956" height="416" alt="Screenshot 2026-09-28 at 19 12 56" src="https://github.com/user-attachments/assets/b90eb0e1-f01f-4416-afdc-cefcb8fb3bc7" />
<img width="956" height="594" alt="Screenshot 2026-09-28 at 19 13 14" src="https://github.com/user-attachments/assets/e6d09a6a-937b-4492-afc1-47d07b95bcc5" />




<img width="693" height="466" alt="Screenshot 2026-09-28 at 19 13 39" src="https://github.com/user-attachments/assets/c2c3d5c2-2436-44a5-86d1-867b35959c9f" />


## Modifikasi 2 — Modified CGAN V2

V2 melanjutkan eksperimen dengan mengubah input conditioning, struktur kedua jaringan, dan learning rate discriminator.

| Komponen | Modified V1 | Modified V2 | Tujuan perubahan |
|---|---|---|---|
| Noise dimension | 128 | 256 | Mengeksplorasi input acak yang lebih besar |
| Label embedding G dan D | 2 | 32 | Memberi kapasitas representasi label yang lebih besar |
| Hidden layer generator | 128 → 256 → 512 → 1024 | 256 → 512 → 1024 | Menguji susunan generator dengan tiga hidden layer dan input yang lebih besar |
| Batch Normalization momentum | Default Keras | 0,8 | Mengubah pembaruan moving statistics pada Batch Normalization |
| Hidden layer discriminator | 512 → 1024 → 1024 → 512 | 512 → 1024 | Mengurangi kedalaman discriminator |
| Discriminator Dropout | 0,3 pada dua layer pertama | 0,4 pada kedua hidden layer | Menguji regularisasi yang lebih kuat |
| Generator learning rate | 0,0001 | 0,0001 | Mempertahankan langkah pembaruan generator |
| Discriminator learning rate | 0,0001 | 0,00005 | Memperlambat pembaruan discriminator relatif terhadap generator |
| Target epoch | 70 | 70 | Mempertahankan target durasi training |

### Generator V2

```text
Noise (256) + label embedding (32)
→ Dense 256 → BatchNorm (momentum 0,8) → LeakyReLU
→ Dense 512 → BatchNorm (momentum 0,8) → LeakyReLU
→ Dense 1024 → BatchNorm (momentum 0,8) → LeakyReLU
→ Dense 784 → Tanh → Reshape (28 × 28 × 1)
```

### Discriminator V2

```text
Flattened image (784) + label embedding (32)
→ Dense 512 → LeakyReLU → Dropout 0,4
→ Dense 1024 → LeakyReLU → Dropout 0,4
→ Dense 1 → Sigmoid
```

### Status V2

Log yang tersimpan berhenti pada **epoch 32 dari target 70**. Catatan eksperimen menyatakan training dihentikan, tetapi notebook belum menyediakan FID maupun visualisasi gambar V2.

Karena itu, kualitas V2 **belum dapat dibandingkan secara kuantitatif** dengan baseline dan V1. Generator loss yang mendekati discriminator loss, atau loss yang terlihat stabil, tidak cukup untuk membuktikan kualitas gambar maupun keberhasilan training.

V2 juga tidak tepat disebut sekadar “model yang lebih kompleks”: noise dan embedding memang membesar, tetapi jumlah hidden layer generator dan discriminator justru berkurang.

## Metode Evaluasi FID

Implementasi evaluasi dalam notebook:

1. Mengambil **1.000 gambar pertama** dari test dataset.
2. Menghasilkan **1.000 gambar sintetis** dengan noise acak dan label yang disampling dari dua kelas.
3. Mengubah ukuran gambar menjadi **299 × 299**, mengubah grayscale menjadi tiga channel, dan menerapkan preprocessing InceptionV3.
4. Mengekstraksi fitur menggunakan **InceptionV3 pretrained ImageNet** dengan global average pooling.
5. Menghitung jarak Fréchet dari rata-rata dan covariance fitur real dan generated menggunakan implementasi NumPy/SciPy.

FID yang lebih rendah menunjukkan distribusi fitur yang lebih dekat dalam protokol evaluasi yang sama. FID bukan accuracy atau persentase kemiripan gambar.

### Keterbatasan Sampling Evaluasi

Test loader menggunakan `shuffle=False`, sementara evaluasi mengambil 1.000 gambar pertama. Dengan urutan direktori per kelas, subset ini dapat didominasi kelas car; komposisinya tidak dijaga agar sesuai dengan label generated yang disampling mendekati seimbang. Karena itu, skor berikut merupakan **hasil eksploratif dari implementasi yang tersimpan**, bukan benchmark dengan sampling kelas yang seimbang.

Evaluasi lanjutan sebaiknya menggunakan sampling real dan generated dengan jumlah yang sama per kelas, noise evaluasi yang tetap, serta FID per kelas atau evaluasi conditioning tambahan. Protokol ini juga tidak otomatis identik dengan implementasi FID pada library lain.

## Ringkasan Hasil

| Model | Epoch pada log | FID tercatat | Penurunan FID terhadap baseline |
|---|---:|---:|---:|
| Baseline CGAN | 40 | 302,88 | — |
| Modified CGAN V1 | 70 | 265,36 | 12,39% |
| Modified CGAN V2 | 32 tercatat; target 70 | Belum tersedia | Belum dapat dihitung |

Angka merupakan output yang tersimpan dalam notebook, bukan hasil training ulang. FID baseline dan V1 dihitung setelah training; pencarian epoch dengan generator loss minimum tidak menyimpan atau memulihkan checkpoint epoch tersebut.

## Temuan Utama

Modified CGAN V1 memperoleh FID lebih rendah dibanding baseline dalam evaluasi yang tercatat. Kombinasi perubahan normalization, regularization, noise dimension, learning rate, dan durasi training memberikan hasil yang menjanjikan untuk eksperimen berikutnya, tetapi belum menunjukkan kontribusi masing-masing komponen secara terpisah.

Eksperimen V2 menunjukkan pentingnya mendokumentasikan percobaan yang belum selesai. Tanpa FID dan sampel gambar, hasil training loss saja tidak cukup untuk menyimpulkan apakah perubahan arsitektur meningkatkan kualitas generasi.

## Pengembangan Selanjutnya

- Memperbaiki sampling FID agar komposisi kelas real dan generated konsisten.
- Menyimpan fixed noise dan label untuk memantau perubahan gambar secara konsisten antar-epoch dan model.
- Menambahkan validation set dan checkpoint selection sebelum evaluasi test akhir.
- Membandingkan model dengan anggaran training yang sama dan beberapa random seed.
- Melakukan ablation study serta mengevaluasi kesesuaian gambar terhadap label conditioning.
- Mengeksplorasi convolutional CGAN untuk memanfaatkan struktur spasial gambar.

## Tools

Python, TensorFlow/Keras, NumPy, SciPy, pandas, Matplotlib, Pillow, dan Google Colab. InceptionV3 digunakan untuk ekstraksi fitur evaluasi; generator dan discriminator dibangun sendiri menggunakan Keras.

## Catatan Notebook Asli

- Tabel ringkasan awal menampilkan 50 epoch untuk baseline dan V1, sedangkan kode dan log menunjukkan **40 dan 70 epoch**. README ini mengikuti kode dan log.
- Salah satu narasi menuliskan baseline FID 301,88; output perhitungan menunjukkan **302,88**.
- Sel contoh generasi dengan `tf.range(0, 10)` menghasilkan label di luar dua kelas yang tersedia. Gunakan `tf.range(10) % NUM_CLASSES` untuk contoh 10 gambar dengan label valid.
- Pemanggilan dataset loader memakai `image_size=IMG_SIZE`; gunakan tuple `(IMG_SIZE, IMG_SIZE)` agar sesuai dengan bentuk argumen ukuran gambar.

## Dataset

https://drive.google.com/drive/folders/1aouIW0R7PwOJoO4BZPpI-wRJdgqaVA6O?usp=sharing
