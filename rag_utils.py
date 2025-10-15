# rag_utils.py
# Ortak yardımcı fonksiyonlar: PDF'ten metin çıkarma, regex ile meta alanları,
# metni parçalara bölme, normalizasyon ve PDF önizleme (PyMuPDF ile).

import os
import re
import unicodedata
from pathlib import Path
from typing import List, Dict, Optional

import fitz  # PyMuPDF
from PIL import Image


# -------------------------
# Metin çıkarımı (PyMuPDF)
# -------------------------
def pdf_text_native(pdf_path: Path) -> str:
    """
    PDF'ten sayfa sayfa metni toplar (PyMuPDF).
    """
    doc = fitz.open(str(pdf_path))
    try:
        texts = []
        for page in doc:
            texts.append(page.get_text() or "")
        return "\n".join(texts).strip()
    finally:
        doc.close()


# -------------------------
# Basit chunking / parçalama
# -------------------------
def chunk_text(text: str, max_len: int = 1200) -> List[str]:
    """
    Uzun metni sabit uzunluklu parçalara böler. (token yerine karakter bazlı basit yaklaşım)
    """
    t = re.sub(r"\s+", " ", text).strip()
    return [t[i:i + max_len] for i in range(0, len(t), max_len)]


# -------------------------
# Metin normalizasyonu
# -------------------------
def normalize_text(s: Optional[str]) -> Optional[str]:
    """
    Aramalarda/filtrelerde karşılaştırmayı kolaylaştırmak için unicode ve case normalizasyonu.
    """
    if s is None:
        return None
    s = s.strip().lower()
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))


# -------------------------
# Regex ile alan yakalama
# -------------------------
def find_invoice_no(text: str) -> Optional[str]:
    """
    'invoice' + opsiyonel # + numara/kimlik kalıbını yakalar (örn: 'Invoice #18843').
    """
    m = re.search(r"(?:invoice\s*#?\s*)([A-Za-z0-9\-_\/]+)", text, re.I)
    return m.group(1) if m else None


def find_total_amount(text: str) -> Optional[str]:
    """
    Para miktarlarını yakalar; en 'büyük/son' toplamı hedeflemek için basit sezgisel.
    Örn: $12,343.63  veya  12,343.63  veya ₺1.234,56 (ikinci biçim için nokta/virgül uyarlaması sınırlı).
    """
    # Önce $/€/₺ formatlı klasik ondalık (noktalı) biçimleri ara
    money = re.findall(r"([\$€₺]?\s?\d{1,3}(?:[,\.\s]\d{3})*(?:[.,]\d{2}))", text)
    if not money:
        return None

    # Heuristik: metinde 'Total' yakınındaki değer önce gelsin
    # 'Total:' satırından sonraki ilk para değeri
    total_match = re.search(r"total[^0-9\$€₺]*([\$€₺]?\s?\d{1,3}(?:[,\.\s]\d{3})*(?:[.,]\d{2}))", text, re.I)
    if total_match:
        return total_match.group(1).strip()

    # Aksi halde, bulunan son para değerini dön (çoğu faturada en sonda yer alır)
    return money[-1].strip()


def find_date(text: str) -> Optional[str]:
    """
    Tarihi yakalar: 'Jan 22 2012', 'Oct 18, 2012', '12/10/2012', '2012-10-18' vb.
    Basit örüntüler; kesin normalize etmiyoruz, yakaladığımız ham string yeter.
    """
    # Örn: Oct 18, 2012 veya Oct 18 2012
    m = re.search(r"\b([A-Z][a-z]{2,9}\s+\d{1,2}(?:,)?\s+\d{4})\b", text)
    if m:
        return m.group(1)

    # Örn: 12/10/2012, 12-10-2012, 12 10 2012
    m = re.search(r"\b(\d{1,2}[\/\-\s]\d{1,2}[\/\-\s]\d{2,4})\b", text)
    if m:
        return m.group(1)

    # Örn: 2012-10-18
    m = re.search(r"\b(\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2})\b", text)
    if m:
        return m.group(1)

    return None


# (Opsiyonel) müşteri adı yakalama: dataset'e göre tutarlılık değişir
def find_customer_guess(text: str) -> Optional[str]:
    """
    Basit 'Bill To:' / 'Ship To:' sezgiseli. Dataset'e göre iyileştirilebilir.
    """
    m = re.search(r"Bill\s*To\s*:?\s*([A-Za-z][A-Za-z .,'\-]+)", text, re.I)
    if m:
        return m.group(1).strip()
    m = re.search(r"Ship\s*To\s*:?\s*([A-Za-z][A-Za-z .,'\-]+)", text, re.I)
    if m:
        return m.group(1).strip()
    return None


# -------------------------
# Metadata çıkarımı
# -------------------------
def extract_meta(text: str, path_str: str = "") -> Dict[str, Optional[str]]:
    """
    PDF'ten çıkarılan ham metinden basit metadata sözlüğü üretir.
    """
    return {
        "invoice_no": find_invoice_no(text),
        "total": find_total_amount(text),
        "date": find_date(text),
        "customer": find_customer_guess(text),
        "path": path_str,
    }


def clean_meta(meta: Dict) -> Dict:
    """
    Chroma gibi sistemler için None'ları ayıklar ve primitive tiplere indirger.
    """
    out = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


# -------------------------
# PDF Önizleme (PyMuPDF)
# -------------------------
def pdf_preview_image(pdf_path: str, dpi: int = 150) -> Optional[Image.Image]:
    """
    İlk sayfayı PIL.Image olarak döndürür. Poppler gerektirmez.
    HF Spaces ve Colab'de güvenli fallback'tir.
    """
    if not pdf_path or not os.path.exists(pdf_path):
        return None
    doc = fitz.open(pdf_path)
    try:
        page = doc.load_page(0)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        mode = "RGBA" if pix.alpha else "RGB"
        img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
        return img
    finally:
        doc.close()
