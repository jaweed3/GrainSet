# GrainSet Rice Quality Inspection - Comprehensive Edge AI Benchmark

This document compiles the benchmark results for deep learning models on the **GrainSet Rice dataset (31,000 images, 8 quality inspection classes)**, comparing the heavy baseline model with lightweight architectures tailored for mobile Edge AI deployment.

---

## 1. Summary Comparison: Baseline vs. Lightweight Models

Evaluation was conducted on the un-augmented **official test set of 3,100 images** across all 8 rice kernel categories.

| Model Architecture | Parameters | Model Size (ONNX FP32) | INT8 Quantized Size | Test Accuracy | Test Macro F1 | Test Weighted F1 | Suitability for Mobile Apps |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **SE-ResNet50 (Baseline)** | 26.1M | 98.4 MB | ~25 MB | **96.42%** | **94.48%** | **96.59%** | ❌ Terlalu berat (latency tinggi, boros baterai) |
| **EfficientNet-B0** | 4.02M | 16.2 MB | ~4.2 MB | **95.71%** | **93.00%** | **95.85%** | ⚠️ Bagus, akurasi tinggi, tapi model 16 MB |
| **MobileNetV4-Small (Google 2024)** | **2.50M** | **9.6 MB** | **2.6 MB** | **95.23%** | **92.19%** | **95.37%** | ⭐️ **REKOMENDASI UTAMA**: 10x lebih kecil, selisih akurasi hanya 1.19%, INT8 hanya 2.6 MB |
| **MobileNetV3-Large** | 4.21M | 17.0 MB | ~4.4 MB | **95.03%** | **92.55%** | **95.22%** | ✔️ Stabil dan kompatibel luas dengan NNAPI |
| **EdgeNeXt-XSmall (Hybrid Transformer)** | **2.15M** | **8.4 MB** | ~2.2 MB | 88.00% | 78.91% | 88.57% | ⚠️ Butuh epoch lebih banyak & dataset besar untuk konvergensi optimal |

---

## 2. Key Research Findings for Thesis (Skripsi)

1. **Akurasi & F1-Score Tetap Solid pada Arsitektur Edge**:
   - `MobileNetV4-Small` mencapai akurasi **95.23%** dan Macro F1 **92.19%**, hanya selisih **1.19%** dibanding baseline `SE-ResNet50` (96.42%), namun dengan jumlah parameter **10 kali lebih sedikit** (2.5M vs 26.1M).
   - `EfficientNet-B0` menyamai angka benchmark paper Nature asli (95.71% vs 95.70%) dengan parameter 6.5x lebih kecil.
2. **Kompresi Model yang Ekstrem (Post-Training Quantization)**:
   - Dari model baseline **98.4 MB**, berhasil dipadatkan menjadi **9.6 MB** (ONNX FP32) dan setelah dynamic INT8 quantization menjadi **2.6 MB** (**97.3% penghematan memori**).
   - Ukuran 2.6 MB sangat ideal untuk di-*embed* langsung ke dalam asset aplikasi mobile (Flutter / Android) tanpa membebani ukuran APK dan memori ponsel petani/pengguna.
3. **Efisiensi Waktu Training**:
   - `SE-ResNet50`: ~140 detik per epoch.
   - `MobileNetV4-Small`: ~19.8 detik per epoch (**7x lebih cepat**).

---

## 3. Detailed Per-Class Breakdown on Test Set (3,100 Images)

### A. SE-ResNet50 (Baseline - 26.1M)
- Test Accuracy: 96.42% | Macro F1: 94.48% | Weighted F1: 96.59%
- Normal: 97.88% F1 | Fusarium: 97.67% F1 | Sprouted: 81.77% F1 | Moldy: 92.67% F1
- Pest Attacked: 95.18% F1 | Broken: 96.62% F1 | Unripe: 94.81% F1 | Impurities: 99.26% F1

### B. EfficientNet-B0 (4.02M)
- Test Accuracy: 95.71% | Macro F1: 93.00% | Weighted F1: 95.85%

### C. MobileNetV4-Small (2.50M - Edge AI Candidate)
- Test Accuracy: 95.23% | Macro F1: 92.19% | Weighted F1: 95.37%

### D. MobileNetV3-Large (4.21M)
- Test Accuracy: 95.03% | Macro F1: 92.55% | Weighted F1: 95.22%

### E. EdgeNeXt-XSmall (2.15M)
- Test Accuracy: 88.00% | Macro F1: 78.91% | Weighted F1: 88.57%

---

## 4. Next Step: Mobile Application Integration

- Format Export: `best_model_int8.onnx` (2.6 MB) & `best_model.tflite`
- Mobile Inference Engine: ONNX Runtime Mobile / TFLite Flutter
- Offline Real-Time Camera Inference & SNI/ISO Rice Quality Assessment
