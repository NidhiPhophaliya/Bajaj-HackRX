from app.utils.pdf_parser import extract_chunks_from_pdfs
import os

pdf_dir = "app/data"
pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith(".pdf")]

extract_chunks_from_pdfs(pdf_files, "app/data/chunks.csv")
print("✅ chunks.csv generated!")
