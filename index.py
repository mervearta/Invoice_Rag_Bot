# index.py
# PDF -> text/meta -> chunks -> embeddings -> Chroma'ya yaz
# Ortam değişkenleri ile yapılandırılabilir:
#   DATA_DIR   : PDF klasörü (default: ./data)
#   CHROMA_DIR : Chroma kalıcı klasör (default: ./chroma_invoices)
#   EMB_MODEL  : sentence-transformers modeli (default: paraphrase-multilingual-MiniLM-L12-v2)
# Kullanım:
#   python index.py            # varsayılanlarla
#   DATA_DIR=/path/to/pdfs python index.py
#   python index.py --limit 100

import os
import uuid
import argparse
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from rag_utils import (
    pdf_text_native,
    chunk_text,
    extract_meta,
    clean_meta,        # rag_utils.py'de var (None temizler, primitive'e çevirir)
)

# ---- Ayarlar (ENV ile override edilebilir) ----
CHROMA_DIR = os.environ.get("CHROMA_DIR", "./chroma_invoices")
EMB_MODEL  = os.environ.get("EMB_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
DATA_DIR   = Path(os.environ.get("DATA_DIR", "./data"))


def index_pdfs(folder: Path, col, limit=None):
    """
    Klasördeki PDF'leri tarar, metin ve metadata çıkarır, chunk'lar ve Chroma'ya ekler.
    """
    pdfs = [p for p in folder.rglob("*.pdf")]
    if limit:
        pdfs = pdfs[:int(limit)]

    ids, docs, metas = [], [], []
    seen = 0
    for p in pdfs:
        seen += 1
        try:
            txt = pdf_text_native(p)
        except Exception as e:
            print(f"[WARN] Metin çıkarılamadı: {p} ({e})")
            continue

        if len(txt) < 30:
            # boş/çok zayıf metni atla (OCR gerekebilir)
            continue

        base_meta = clean_meta(extract_meta(txt, path_str=str(p)))
        # chunk'la ve kuyruğa ekle
        for idx, ch in enumerate(chunk_text(txt)):
            ids.append(str(uuid.uuid4()))
            docs.append(ch)
            m = dict(base_meta)
            m["chunk"] = int(idx)  # primitive tip
            metas.append(m)

    if docs:
        # güvenlik: tüm metadatalar primitive mi?
        for m in metas:
            for k, v in list(m.items()):
                if v is None:
                    del m[k]
                elif not isinstance(v, (str, int, float, bool)):
                    m[k] = str(v)

        col.add(ids=ids, documents=docs, metadatas=metas)

    return seen, len(docs)


def main():
    # Klasör kontrolü
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"DATA_DIR bulunamadı: {DATA_DIR}")

    print(f"[INFO] DATA_DIR   : {DATA_DIR}")
    print(f"[INFO] CHROMA_DIR : {CHROMA_DIR}")
    print(f"[INFO] EMB_MODEL  : {EMB_MODEL}")

    # Argümanlar
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="İndekslenecek PDF sayısını sınırla (opsiyonel)")
    args = parser.parse_args()

    # Chroma client & koleksiyon
    client  = chromadb.PersistentClient(path=CHROMA_DIR)
    embedfn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMB_MODEL)

    # Temiz koleksiyon (yeniden oluştur)
    try:
        client.delete_collection("invoices")
        print("[INFO] Eski 'invoices' koleksiyonu silindi.")
    except Exception:
        pass

    col = client.create_collection("invoices", embedding_function=embedfn)
    print("[INFO] Yeni 'invoices' koleksiyonu oluşturuldu.")

    # İndeksleme
    seen, added_chunks = index_pdfs(DATA_DIR, col, limit=args.limit)
    print(f"[DONE] Görülen PDF: {seen} | Chroma'ya eklenen chunk: {added_chunks}")


if __name__ == "__main__":
    main()
