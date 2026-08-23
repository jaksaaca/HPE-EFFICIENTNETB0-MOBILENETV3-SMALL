# Head Pose Estimation (Regression)

Project ini bertujuan untuk membangun model Head Pose Estimation murni berbasis regresi. Model hanya memprediksi nilai tiga sudut kontinu: **pitch**, **yaw**, dan **roll** (dalam derajat). Tidak ada klasifikasi kategori arah kepala (seperti Front, Left, Right) yang digunakan dalam proses training maupun evaluasi model.

## Model

Terdapat dua model yang digunakan dan dibandingkan:
1. **EfficientNetB0**: Model utama dengan performa ekstraksi fitur tinggi.
2. **MobileNetV3-Small**: Model pembanding *lightweight* untuk mengevaluasi trade-off efisiensi pada perangkat dengan memori terbatas.

## Dataset & Validasi Lintas Dataset

Project ini menggunakan skenario evaluasi dunia nyata berupa **Validasi Lintas Dataset (Cross-Dataset Validation)**:
- **Training**: Seluruh pasangan citra dan anotasi yang valid dari dataset **300W-LP** digunakan untuk melatih model.
- **Validasi**: Seluruh pasangan citra dan anotasi yang valid dari dataset **AFLW2000-3D** digunakan untuk menghitung loss validasi dan performa model (tidak digunakan untuk memperbarui bobot model).

**Tidak ada**:
- Pembatasan dataset (seperti subset 30.000 data)
- Splitting train-test (misal 80:20 dari satu sumber dataset)
- Klasifikasi atau perhitungan tingkat akurasi klasifikasi (accuracy, precision, recall, F1-score)

## Strategi Training Final
- Seluruh training dilakukan secara langsung selama **15 Epoch** (bukan eksperimen awal atau trial).
- **Epoch 1-3**: Seluruh backbone dibekukan (frozen). Hanya regression head yang dilatih.
- **Epoch 4-15**: Blok terakhir backbone dibuka (unfrozen) untuk *fine-tuning*.
- **Early Stopping**: Tidak ada *early stopping*. Checkpoint model terbaik disimpan berdasarkan Validation Mean MAE terendah yang diperoleh di antara epoch 1 hingga 15.

## Metrik
### 1. Metrik Regresi
- **MAE Pitch**
- **MAE Yaw**
- **MAE Roll**
- **Mean MAE** (Rata-rata dari ketiga MAE di atas. Lebih rendah = lebih baik).
### 2. Metrik Lightweight
- Total Parameters & Trainable Parameters
- Model Size (MB)
- FLOPs & GFLOPs
- Inference Latency (ms) & FPS
- Peak GPU Memory (MB)

## Cara Penggunaan

### 1. Instalasi
```bash
pip install -r requirements.txt
```

### 2. Training
Jalankan training secara berurutan. (Sebaiknya jalankan satu per satu jika menggunakan satu GPU, untuk menghindari Out of Memory).

```bash
# Training EfficientNetB0
python src/train.py --model efficientnet --epochs 15 --batch-size 16

# Training MobileNetV3-Small
python src/train.py --model mobilenet --epochs 15 --batch-size 16
```
*(Ubah `--batch-size` menjadi 8 atau 32 sesuai kapasitas memori GPU Anda.)*

### 3. Melanjutkan Training (Resume)
Jika training terhenti di tengah jalan (misal karena listrik padam atau memori habis), Anda bisa melanjutkannya:
```bash
python src/train.py --model efficientnet --epochs 15 --batch-size 16 --resume outputs/models/efficientnet_last.pth
```
Resume akan otomatis memuat best validation MAE sebelumnya dan status optimizer.

### 4. Evaluasi & Benchmark
Evaluasi model (menggunakan best checkpoint) dan hitung benchmark kecepatannya:
```bash
python src/evaluate.py --model efficientnet
python src/evaluate.py --model mobilenet
```

### 5. Komparasi Akhir
Bandingkan performa kedua model (setelah tahap evaluasi selesai):
```bash
python src/compare.py
```
Output komparasi dapat dilihat di file CSV, Markdown, dan berbagai plot yang disimpan.

### 6. Demo Webcam
Jalankan live inferensi dari kamera Anda.
```bash
python src/webcam.py --model efficientnet
```
**Catatan**: Terdapat perbedaan perhitungan FPS di sini. FPS *benchmark* pada tahap evaluasi hanya menghitung kecepatan *forward pass* model murni. Sedangkan FPS *webcam* adalah FPS end-to-end yang mencakup frame capture kamera, face detection menggunakan Haar Cascade, pre-processing, inference, dan penggambaran (overlay) UI ke frame.

## Lokasi Output
Seluruh hasil otomatis disimpan pada struktur folder berikut:
- **Model Checkpoints**: `outputs/models/` (best, last, dan log tiap epoch)
- **Log & Reports**: `outputs/reports/` (CSV logs, benchmark logs, dataset manifest)
- **Plots**: `outputs/plots/` (grafik training curve, comparison bar charts)
