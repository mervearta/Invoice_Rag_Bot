# app.py
# Gradio arayüzü: RAG ile fatura soruları + PDF önizleme
# Ortam değişkenleri:
#   CHROMA_DIR : ./chroma_invoices (varsayılan)
#   EMB_MODEL  : sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (varsayılan)
#   LLM_MODEL  : google/flan-t5-base (varsayılan)

import os
import re
from typing import List, Tuple, Optional

import gradio as gr
import chromadb
from chromadb.utils import embedding_functions
from transformers import pipeline

from rag_utils import pdf_preview_image  # PDF ilk sayfa görseli için

# ---- Ayarlar ----
CHROMA_DIR = os.environ.get("CHROMA_DIR", "./chroma_invoices")
EMB_MODEL  = os.environ.get("EMB_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
LLM_MODEL  = os.environ.get("LLM_MODEL", "google/flan-t5-base")

# ---- Chroma & Embedding ----
client  = chromadb.PersistentClient(path=CHROMA_DIR)
embedfn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMB_MODEL)
col     = client.get_or_create_collection("invoices", embedding_function=embedfn)

# ---- LLM ----
gen = pipeline("text2text-generation", model=LLM_MODEL, device_map="auto")


def build_prompt(question: str, ctx: List[Tuple[str, dict]]) -> str:
    """
    RAG prompt'u: yalnızca sağlanan bağlamdan cevapla; bilinmiyorsa söyle.
    """
    bullets = []
    for _, m in ctx[:3]:
        bullets.append(
            f"- Invoice:{m.get('invoice_no')} | Date:{m.get('date')} | Total:{m.get('total')}"
        )
    return (
        "Answer ONLY based on the invoice facts below. "
        "If unknown, say 'I don't know'. Reply concisely in English.\n\n"
        + "\n".join(bullets)
        + f"\n\nQuestion: {question}\nAnswer:"
    )


def answer(question: str, k: int = 5, where: Optional[dict] = None, max_new_tokens: int = 96):
    """
    Top-k retrieval + LLM cevabı döndürür.
    """
    res = col.query(query_texts=[question], n_results=k, where=where or {})
    docs  = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    ctx = list(zip(docs, metas))
    if not ctx:
        return "I don't know.", []

    out = gen(
        build_prompt(question, ctx),
        max_new_tokens=max_new_tokens,
        do_sample=False
    )[0]["generated_text"].strip()
    return out, ctx


def _find_pdf_in_ctx_by_invoice(invoice_no: str, ctx: List[Tuple[str, dict]]) -> Optional[str]:
    # 1) metadata'da tam eşleşme
    for _, m in ctx:
        if str(m.get("invoice_no")) == str(invoice_no):
            return m.get("path")
    # 2) dosya adında geçme (bazı setlerde isimde de bulunur)
    for _, m in ctx:
        fname = os.path.basename(m.get("path", ""))
        if invoice_no and invoice_no in fname:
            return m.get("path")
    return None


def find_pdf_for_invoice(invoice_no: str, ctx: List[Tuple[str, dict]]) -> Optional[str]:
    """
    Önce bağlamda ara; yoksa Chroma'da doğrudan where ile bul.
    """
    # bağlamda dene
    hit = _find_pdf_in_ctx_by_invoice(invoice_no, ctx)
    if hit:
        return hit

    # doğrudan where ile dene
    try:
        res = col.query(
            query_texts=[f"invoice {invoice_no}"],
            n_results=1,
            where={"invoice_no": str(invoice_no)}
        )
        md = res.get("metadatas", [[]])[0]
        if md:
            return md[0].get("path")
    except Exception:
        pass
    return None


def chat_with_pdf(user_q: str, invoice_no: str):
    """
    Gradio callback: soru + isteğe bağlı invoice numarası.
    Cevap + kaynak listesi + PDF önizleme döndürür.
    """
    try:
        # Soru metninden invoice # yakalamayı dene (UI alanı boşsa)
        if not invoice_no and user_q:
            m = re.search(r"(?:invoice\s*#?\s*)([A-Za-z0-9\-_\/]+)", user_q, re.I)
            if m:
                invoice_no = m.group(1)

        # Filtre: varsa sadece invoice_no'yu uygula
        where = {"invoice_no": str(invoice_no)} if invoice_no else None

        # RAG cevabı
        ans, ctx = answer(user_q or "invoice info", k=7, where=where)
        if not ctx:
            return "No related invoices found.", None

        # Doğru PDF'i tespit et
        pdf_path = find_pdf_for_invoice(str(invoice_no), ctx) if invoice_no else None
        if not pdf_path:
            pdf_path = ctx[0][1].get("path")

        # Kaynakları listele
        src_lines = []
        for _, m in ctx[:5]:
            src_lines.append(
                f"- **{os.path.basename(m.get('path',''))}** "
                f"(Inv:{m.get('invoice_no')}, Date:{m.get('date')}, Total:{m.get('total')})"
            )

        # PDF linki (Spaces'ta dosya yolu erişimi demo içindir)
        pdf_link = f"[📄 View PDF File]({pdf_path})" if pdf_path else ""
        full_md = f"**Answer:** {ans}\n\n{pdf_link}\n\n---\n**Sources:**\n" + "\n".join(src_lines)

        # Önizleme görseli
        preview = pdf_preview_image(pdf_path) if pdf_path else None
        return full_md, preview

    except Exception as e:
        return f"⚠️ Error: {e}", None


# ---- Gradio Arayüzü ----
demo = gr.Interface(
    fn=chat_with_pdf,
    inputs=[
        gr.Textbox(label="Question", value="What is the total and date for invoice #18843?"),
        gr.Textbox(label="Invoice No (optional)")
    ],
    outputs=[
        gr.Markdown(label="Answer"),
        gr.Image(label="PDF Preview")
    ],
    title="📄 Invoice & Offer RAG Chatbot",
    description="Ask about invoices; the app retrieves from a Chroma index and shows the related PDF preview."
)

if __name__ == "__main__":
    # HF Spaces'ta standart launch yeterlidir.
    demo.queue().launch()
