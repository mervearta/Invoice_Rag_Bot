# 📄 Invoice & Offer RAG Chatbot

Bu proje, PDF faturalardan bilgi çıkarıp anlamlandıran **RAG (Retrieval-Augmented Generation)** tabanlı bir chatbot’tur.  
Kullanıcıdan gelen fatura numarasına göre toplam tutarı yanıtlayabilir, ayrıca ilgili PDF’in **önizlemesini** de Gradio arayüzünde gösterir.

---

## 📊 Veri Seti

Projede kullanılan PDF faturalar, **[Sample-Pdf-invoices](https://github.com/femstac/Sample-Pdf-invoices)** adlı açık kaynak veri setinden alınmıştır.  
Bu repo içerisinde **1000’den fazla PDF fatura** yer almaktadır. Her PDF dosyasında farklı müşterilere ait fatura bilgileri (fatura numarası, tarih, toplam tutar vb.) bulunmaktadır.

Proje kapsamında bu veri setinden **50 adet PDF** seçilerek **Chroma vektör veritabanına** eklenmiş ve indekslenmiştir.  
Uygulama, bu veritabanındaki faturaları sorgulayarak belirli bir **fatura numarasına** ait **toplam tutarı** bulur ve kullanıcıya döndürür.

---

## ⚙️ Yöntem ve Mimarisi

Uygulama, **Retrieval-Augmented Generation (RAG)** mimarisi üzerine kuruludur.  
Aşağıdaki adımlar takip edilmiştir:

1. **Veri Hazırlama ve İndeksleme**
   - PDF dosyalarından metin çıkarımı için **PyMuPDF** (`fitz`) kullanıldı.  
   - Regex yardımıyla `invoice_no`, `total`, `date` gibi temel alanlar metinden ayrıştırıldı.  
   - Her fatura metni belirli uzunluklarda parçalara (chunk) bölünerek **ChromaDB** koleksiyonuna eklendi.  
   - Her parçaya ait embedding’ler, çok dilli bir model ile oluşturuldu.

2. **Embedding ve LLM Modelleri**
   - **Embedding modeli:**
     ```
     sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
     ```
     Bu model, Türkçe ve İngilizce dahil olmak üzere çok dilli sorgular için uygundur.
   - **LLM modeli:**
     ```
     google/flan-t5-base
     ```
     Kısa ve doğru metin üretimi için kullanılmıştır.

3. **RAG Mantığı**
   - Kullanıcı, örneğin “What is the total and date for invoice #18843?” sorusunu sorduğunda, sistem Chroma’dan en alakalı döküman parçalarını getirir.  
   - Bu parçalar, `Flan-T5` modeline bağlam olarak verilerek nihai yanıt üretilir.  
   - Cevapla birlikte, ilgili PDF’in **önizlemesi** de Gradio arayüzünde gösterilir.

---

## 🧠 Ortam Değişkenleri (Environment Variables)

Uygulama aşağıdaki ortam değişkenleriyle yapılandırılabilir:

```python
CHROMA_DIR = os.environ.get("CHROMA_DIR", "./chroma_invoices")
EMB_MODEL  = os.environ.get("EMB_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
LLM_MODEL  = os.environ.get("LLM_MODEL", "google/flan-t5-base")
