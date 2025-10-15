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
