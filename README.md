# 📄 Invoice & Offer RAG Chatbot

Bu proje, PDF faturalardan bilgi çıkarıp anlamlandıran **RAG (Retrieval-Augmented Generation)** tabanlı bir chatbot’tur.  
Kullanıcıdan gelen fatura numarasına göre toplam tutarı ve tarihi yanıtlayabilir, ayrıca ilgili PDF’in **önizlemesini** de Gradio arayüzünde gösterir.

---

## 📊 Veri Seti

Bu projede, **[Sample-Pdf-invoices](https://github.com/femstac/Sample-Pdf-invoices)** deposundaki **1000+ PDF** fatura kullanılmıştır.  
Her dosyada farklı müşterilere ait **fatura numarası, tarih ve toplam tutar** gibi alanlar yer almaktadır.  
Bu veri setinden **50 PDF** seçilerek **Chroma** vektör veritabanına **indekslenmiştir**.  
Uygulama, girilen **fatura numarasına** göre ilgili faturayı bulur ve **toplam tutarı** (ve istenirse tarihi) döndürür.

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


## 🧾 Örnek Sorgular (Test İçin)

Aşağıdaki fatura numaralarını test etmek için kullanabilirsiniz:

```
36258  
36259  
40100  
4820  
15978
```

💬 **Örnek sorgular:**
```
What is the total and date for invoice #36258?
What is the total for invoice #40100?
15978.
```
<img width="1904" height="919" alt="5" src="https://github.com/user-attachments/assets/c6933759-d2bb-4496-89db-9ed35e01c217" />



Bu sorguları Hugging Face Spaces üzerinde test edebilirsiniz:  
👉 [**Invoice & Offer RAG Chatbot (Hugging Face)**](https://huggingface.co/spaces/MerveBaydar/invoice-rag-bot)

