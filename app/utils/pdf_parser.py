import fitz  # PyMuPDF
import uuid
import pandas as pd
import os

def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def extract_chunks_from_pdfs(pdf_paths, output_csv_path):
    all_chunks = []

    for file_path in pdf_paths:
        doc = fitz.open(file_path)
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            chunks = chunk_text(text)
            for idx, chunk in enumerate(chunks):
                all_chunks.append({
                    "chunk_id": str(uuid.uuid4()),
                    "source_doc": os.path.basename(file_path),
                    "page": page_num,
                    "chunk_index": idx,
                    "text": chunk
                })

    df['text'] = df['text'].apply(lambda t: t[:1000])            
    df = pd.DataFrame(all_chunks)
    df.to_csv(output_csv_path, index=False)
    return df
