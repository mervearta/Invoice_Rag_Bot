# 📄 Invoice & Offer RAG Chatbot

Bu proje, PDF faturalardan bilgi çıkarıp anlamlandıran **RAG (Retrieval-Augmented Generation)** tabanlı bir chatbot’tur.  
Kullanıcıdan gelen fatura numarasına göre toplam tutarı ve tarihi yanıtlayabilir, ayrıca ilgili PDF’in **önizlemesini** de Gradio arayüzünde gösterir.

---

## 🚀 Kurulum ve Çalıştırma Adımları

1. **Gerekli bağımlılıkları yükleyin:**
   ```bash
   pip install -r requirements.txt
   ```

2. **PDF dosyalarının bulunduğu klasörü belirleyin (örnek: `data/`):**
   ```bash
   export DATA_DIR=./data
   ```

3. **PDF’leri indeksleyin:**
   ```bash
   python index.py
   ```

4. **Gradio arayüzünü başlatın:**
   ```bash
   python app.py
   ```

5. **Tarayıcıda açılan arayüz üzerinden şu tür sorular sorabilirsiniz:**
   ```
   What is the total and date for invoice #18843?
   ```

---

## 🖥️ Arayüz Özellikleri

- Fatura numarasına göre toplam ve tarih sorgulama  
- PDF önizlemesi  
- Kaynak fatura bilgilerini Markdown formatında gösterim  
- Gradio tabanlı basit ve etkileşimli web arayüzü  

---

## 🌐 Demo

Proje Hugging Face Spaces üzerinde yayınlanmıştır:  
👉 [**Invoice & Offer RAG Chatbot (Hugging Face)**](https://huggingface.co/spaces/MerveBaydar/invoice-rag-bot)

---

## 📦 Kullanılan Teknolojiler

| Teknoloji | Açıklama |
|------------|-----------|
| **PyMuPDF** | PDF metin çıkarımı |
| **Regex (re)** | Fatura numarası, tarih, toplam tutar tespiti |
| **SentenceTransformer** | Çok dilli embedding üretimi |
| **ChromaDB** | Vektör veritabanı |
| **Flan-T5 (Google)** | Metin tabanlı yanıt üretimi |
| **Gradio** | Web arayüzü ve etkileşimli chatbot |
| **Pillow (PIL)** | PDF önizleme görseli oluşturma |

---

## 🧩 Dosya Yapısı

```
├── app.py             # Gradio arayüzü ve chatbot mantığı
├── index.py           # PDF indeksleme işlemleri
├── rag_utils.py       # Yardımcı fonksiyonlar (regex, PDF parse, önizleme)
├── requirements.txt   # Gerekli kütüphaneler
├── README.md          # Proje açıklaması
└── chroma_invoices/   # Chroma vektör veritabanı
```

---

## 🎯 Sonuç

Bu proje, **RAG mimarisiyle PDF tabanlı bilgi çıkarımı ve sorgulama** sürecini uçtan uca göstermektedir.  
Kullanıcı, fatura numarasını girerek doğrudan toplam tutarı ve tarihi alabilir.  
Ayrıca sistem, ilgili PDF’in görsel önizlemesini sunarak yanıtın kaynağını da açık biçimde gösterir.
