# PROJECT CONTEXT

Ini adalah *source of truth* dari project penelitian Head Pose Estimation berbasis regresi.

## KEPUTUSAN FINAL PENELITIAN (MUTLAK)
1. **Task Hanya Regresi**: Output model hanyalah 3 nilai kontinu: pitch, yaw, dan roll dalam derajat.
2. **Tidak Ada Klasifikasi Arah**: Model tidak mengklasifikasi pose ke dalam arah (front, left, right, up, down). 
3. **Tidak Ada Metrik Klasifikasi**: Dilarang menggunakan accuracy, precision, recall, F1-score, dan confusion matrix.
4. **Dataset Penuh**: Full data valid 300W-LP mutlak digunakan untuk training. Full data valid AFLW2000-3D digunakan untuk *cross-dataset validation*.
5. **Tanpa Subset/Batasan**: Tidak ada limitasi 30.000 sampel. Tidak ada argument `--max-train-samples`. Tidak ada dataset split 80:20.
6. **Training 15 Epoch (One-Shot)**: Model langsung dilatih 15 epoch tanpa eksperimen awal. Epoch 1-3 melatih regression head. Epoch 4-15 membuka blok terakhir backbone.
7. **Best Checkpoint**: Dipilih murni dari epoch dengan **Validation Mean MAE** terendah. Tidak ada early stopping.
8. **Checkpoint Disimpan per Epoch**: Memungkinkan fitur `--resume` jika proses training berhenti di tengah jalan.
9. **Fokus Evaluasi Evaluasi Lightweight**: Wajib mengevaluasi total/trainable params, ukuran model, FLOPs/GFLOPs, latency, FPS benchmark, dan GPU memory.
10. **Webcam Demo**: `webcam.py` hanya digunakan sebagai sarana demonstrasi hasil nilai regresi.
11. **Konversi Kategori Nanti Saja**: Konversi nilai regresi menjadi kategori arah kepala akan dibuat secara terpisah pada tahap pengembangan aplikasi web dan bukan pada evaluasi performa model ini.

---

## AI HANDOFF SUMMARY

**Tujuan Project**: Membangun model Head Pose Estimation murni regresi (pitch, yaw, roll) tanpa aktivasi akhir.
**Model**: EfficientNetB0 (utama) vs MobileNetV3-Small (lightweight pembanding).
**Dataset**: Full 300W-LP valid untuk Training. Full AFLW2000-3D valid untuk Cross-Dataset Validation.
**Konfigurasi Training**: 15 epoch, batch size 16 (default), AdamW, SmoothL1Loss(beta=1.0), ReduceLROnPlateau. Epoch 1-3 backbone beku, epoch 4-15 buka blok terakhir.
**Metrik Regresi**: MAE Pitch, MAE Yaw, MAE Roll, Mean MAE (lebih rendah lebih baik).
**Metrik Lightweight**: Params, Size MB, GFLOPs, Latency ms, FPS, Peak GPU Mem.
**Checkpoint**: Disimpan per epoch, best diambil dari Val Mean MAE terendah. Mendukung argumen `--resume`.
**Status Project**: Kode telah selesai diaudit dan diperbarui sesuai seluruh instruksi final. Siap dijalankan oleh user.
**Command Berikutnya**: User dapat langsung memulai `python src/train.py --model efficientnet`
**Keputusan Mutlak (Dilarang Diubah)**: Jangan tambahkan klasifikasi, confusion matrix, F1-score, accuracy. Jangan pecah 300W-LP. Gunakan AFLW2000-3D hanya sebagai evaluasi, jangan untuk backward pass.
