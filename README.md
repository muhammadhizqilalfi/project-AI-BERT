---

title: BERT Sentiment API
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
--------------

# FastAPI + IndoBERT Sentiment Analysis API

## 📌 Penjelasan Project

Project **Sentiment Analyzer** ini merupakan Proyek UAS mata kuliah **Kecerdasan Artifisial** yang berfokus pada pengembangan model **Artificial Intelligence** untuk melakukan **analisis sentimen tweet berbahasa Indonesia** dari platform **X (Twitter)**.

Model yang digunakan adalah **IndoBERT (indobenchmark/indobert-base-p1)** yang telah di-*fine-tune* untuk mengklasifikasikan sentimen ke dalam tiga kelas, yaitu **positif**, **negatif**, dan **netral**. Sistem ini di-*deploy* dalam bentuk **REST API** menggunakan **FastAPI** dan dikemas dengan **Docker**, serta diintegrasikan dengan aplikasi web sebagai antarmuka pengguna.

---

## 🏷️ Penjelasan Label Sentimen

Model mengklasifikasikan teks ke dalam tiga label berikut:

| Label | Keterangan                                      |
| ----- | ----------------------------------------------- |
| 0     | Negatif – opini atau pernyataan bernada negatif |
| 1     | Netral – opini bersifat objektif atau ambigu    |
| 2     | Positif – opini atau pernyataan bernada positif |

---

## 🗂️ Dataset

* Dataset berupa kumpulan tweet berbahasa Indonesia dari platform X
* Jumlah data: **5088 baris**
* Format: **CSV**
* Sumber dataset: Repository publik GitHub

🔗 Link Dataset:


---

## ⚙️ Tahapan Preprocessing Data

1. Casefolding
2. Menghapus link dan simbol
3. Tokenisasi
4. Menghapus stopword
5. Normalisasi kata
6. Vektorisasi teks

---

## 🧠 Model & Training

* Pretrained Model: **IndoBERT (indobenchmark/indobert-base-p1)**
* Jumlah kelas: 3 (Negatif, Netral, Positif)
* Max sequence length: 128
* Epoch: 3
* Batch size: 16
* Learning rate: 2e-5
* Weight decay: 0.01
* Evaluasi: Accuracy & Macro F1-Score
* Accuracy akhir: ±82.5%
* Macro F1-Score: ±78.7%

Model terbaik disimpan dan digunakan untuk prediksi serta deployment.

---

## 🚀 Deployment & Arsitektur Sistem

### Backend

* FastAPI
* Docker Container
* Hugging Face Spaces

### Frontend

* Next.js
* Hosting: Vercel

### Alur Sistem

1. User memasukkan teks tweet melalui web
2. Frontend mengirim request ke API
3. Backend memproses teks menggunakan model IndoBERT
4. Hasil sentimen ditampilkan ke user

---

## 🖼️ Gambar Web

```md
![Tampilan Web](assets/web-preview.png)
```

---

## ▶️ Contoh Demo

Contoh request ke API:

```json
POST /predict
{
  "text": "Pelayanan aplikasi ini sangat membantu"
}
```

Contoh response:

```json
{
  "label": "positif",
  "confidence": 0.87
}
```

---

## 🔗 Link Terkait

* Dataset: 
* Repository GitHub: 
* Demo Web: 
* API Endpoint: 

---

## 👥 Anggota Kelompok

| Nama                    | NIM           | Kontribusi                             |
| ----------------------- | ------------- | -------------------------------------- |
| Muhammad Hizqil Alfi    | 2308107010046 | Backend, model, integrasi & deployment |
| Riyan Hadi Samudra      | 2308107010068 | Persiapan dataset                      |
| Muhammad Caesar Aidarus | 2308107010072 | Frontend                               |
| Razian Sabri            | 2308107010050 | Preprocessing dataset                  |

---

## 📌 Kesimpulan

Model IndoBERT yang telah di-*fine-tune* mampu melakukan analisis sentimen tweet berbahasa Indonesia secara efektif dan stabil. Sistem ini berhasil mengklasifikasikan sentimen negatif, netral, dan positif dengan performa yang baik, serta layak diimplementasikan dalam aplikasi web untuk prediksi sentimen secara *real-time*. Peningkatan performa masih dapat dilakukan terutama pada kelas sentimen netral melalui penambahan data dan optimasi lanjutan.

---

✨ *Project ini dikembangkan sebagai bagian dari Proyek UAS Kecerdasan Artifisial*
