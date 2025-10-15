# 📄 Invoice & Offer RAG Chatbot

Bu proje, PDF faturaları okuyup anlamlandıran **RAG (Retrieval-Augmented Generation)** tabanlı bir chatbot’tur.  
1000 faturadan seçilen 50 tanesini **Chroma vektör veritabanına** yükleyerek, kullanıcıdan gelen **fatura numarasına göre toplam tutar** sorgularını yanıtlar.  
Ayrıca ilgili PDF’in **önizlemesini** de Gradio arayüzünde gösterir.

---

## 🚀 Özellikler
- **PyMuPDF** ile metin çıkarımı  
- **Regex** ile `invoice_no`, `date`, `total` alanlarının yakalanması  
- **SentenceTransformer** tabanlı çok dilli embedding  
- **Chroma** vektör veritabanı  
- **Flan-T5** modeli ile yanıt üretimi  
- **Gradio** arayüzü (PDF önizleme dahil)

---

## ⚙️ Kurulum
```bash
pip install -r requirements.txt

📘 Veri Seti ve Model Bilgisi

Bu projede, Sample PDF Invoices
 deposunda yer alan 1000+ PDF faturadan oluşan açık kaynak veri seti kullanılmıştır.
Veri setinde farklı müşterilere ait çeşitli fatura örnekleri bulunmaktadır.

Uygulama, her müşteriye ait fatura numaraları üzerinden sorgulama yaparak, ilgili faturanın toplam tutarını otomatik olarak döndürmektedir.
Bu işlemi gerçekleştirmek için veri ön işleme, indeksleme ve sorgu yanıtlama adımlarını içeren bir RAG (Retrieval-Augmented Generation) mimarisi kullanılmıştır.

Veri setinin tamamı içinden 50 adet fatura seçilerek Chroma vektör veritabanına yüklenmiş ve indekslenmiştir.
Bu sayede sistem, kullanıcıdan gelen bir “invoice number” sorgusuna göre en ilgili fatura verisini hızlıca bulup yanıtlayabilmektedir.
