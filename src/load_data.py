from pathlib import Path
import fitz  # PyMuPDF


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract text from a PDF file using PyMuPDF.
    Returns all page text joined into one string.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    all_pages_text: list[str] = []

    with fitz.open(pdf_path) as doc:
        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            raw_text = page.get_text("text")
            text = str(raw_text)

            if text:
                cleaned_text = text.strip()
                if cleaned_text:
                    all_pages_text.append(
                        f"\n--- Page {page_index + 1} ---\n{cleaned_text}"
                    )

    return "\n".join(all_pages_text).strip()


def extract_text_from_txt(txt_path: Path) -> str:
    """
    Read text from a .txt transcript file.
    """
    if not txt_path.exists():
        raise FileNotFoundError(f"Transcript file not found: {txt_path}")

    return txt_path.read_text(encoding="utf-8").strip()


def load_all_pdfs(pdf_folder: Path) -> dict[str, str]:
    """
    Load all PDFs from a folder.
    Returns dictionary: {filename: extracted_text}
    """
    if not pdf_folder.exists():
        raise FileNotFoundError(f"PDF folder not found: {pdf_folder}")

    pdf_texts: dict[str, str] = {}

    for pdf_file in sorted(pdf_folder.glob("*.pdf")):
        print(f"Loading PDF: {pdf_file.name}")
        pdf_texts[pdf_file.name] = extract_text_from_pdf(pdf_file)

    return pdf_texts


def load_all_transcripts(transcript_folder: Path) -> dict[str, str]:
    """
    Load all transcript .txt files from a folder.
    Returns dictionary: {filename: text}
    """
    if not transcript_folder.exists():
        raise FileNotFoundError(f"Transcript folder not found: {transcript_folder}")

    transcript_texts: dict[str, str] = {}

    for txt_file in sorted(transcript_folder.glob("*.txt")):
        print(f"Loading transcript: {txt_file.name}")
        transcript_texts[txt_file.name] = extract_text_from_txt(txt_file)

    return transcript_texts


def clean_text(text: str) -> str:
    """
    Basic text cleanup:
    - normalize line endings
    - remove excessive blank lines
    - remove repeated spaces
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = []
    previous_blank = False

    for line in text.split("\n"):
        cleaned_line = " ".join(line.split()).strip()

        if cleaned_line == "":
            if not previous_blank:
                lines.append("")
            previous_blank = True
        else:
            lines.append(cleaned_line)
            previous_blank = False

    return "\n".join(lines).strip()


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """
    Split text into overlapping chunks based on character count.
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    cleaned = clean_text(text)

    if not cleaned:
        return []

    chunks: list[str] = []
    start = 0
    text_length = len(cleaned)

    while start < text_length:
        end = start + chunk_size
        chunk = cleaned[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start += chunk_size - overlap

    return chunks


def build_chunk_records(
    pdf_documents: dict[str, str],
    transcript_documents: dict[str, str],
    chunk_size: int = 800,
    overlap: int = 150,
) -> list[dict]:
    """
    Convert loaded documents into structured chunk records with metadata.
    """
    records: list[dict] = []

    for file_name, text in pdf_documents.items():
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

        for chunk_index, chunk in enumerate(chunks):
            records.append(
                {
                    "source_type": "pdf",
                    "file_name": file_name,
                    "chunk_index": chunk_index,
                    "text": chunk,
                }
            )

    for file_name, text in transcript_documents.items():
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

        for chunk_index, chunk in enumerate(chunks):
            records.append(
                {
                    "source_type": "transcript",
                    "file_name": file_name,
                    "chunk_index": chunk_index,
                    "text": chunk,
                }
            )

    return records


def print_preview(documents: dict[str, str], source_name: str, preview_length: int = 500) -> None:
    """
    Print short preview of loaded documents.
    """
    print(f"\n===== {source_name} PREVIEW =====")

    if not documents:
        print("No documents found.")
        return

    for file_name, text in documents.items():
        print(f"\nFile: {file_name}")
        print(f"Characters extracted: {len(text)}")
        print("Preview:")
        print(text[:preview_length])
        print("\n" + "=" * 60)


def print_chunk_preview(chunk_records: list[dict], preview_count: int = 5, preview_length: int = 300) -> None:
    """
    Print a preview of the first few chunks.
    """
    print("\n===== CHUNK PREVIEW =====")
    print(f"Total chunks created: {len(chunk_records)}")

    for record in chunk_records[:preview_count]:
        print("\nSource type:", record["source_type"])
        print("File name:", record["file_name"])
        print("Chunk index:", record["chunk_index"])
        print("Chunk length:", len(record["text"]))
        print("Chunk preview:")
        print(record["text"][:preview_length])
        print("\n" + "=" * 60)


def main() -> None:
    print("Script started")

    project_root = Path(__file__).resolve().parent.parent
    pdf_folder = project_root / "data" / "pdfs"
    transcript_folder = project_root / "data" / "transcripts"

    print(f"PDF folder: {pdf_folder}")
    print(f"Transcript folder: {transcript_folder}")

    pdf_documents = load_all_pdfs(pdf_folder)
    transcript_documents = load_all_transcripts(transcript_folder)

    print_preview(pdf_documents, "PDF")
    print_preview(transcript_documents, "TRANSCRIPT")

    chunk_records = build_chunk_records(
        pdf_documents=pdf_documents,
        transcript_documents=transcript_documents,
        chunk_size=800,
        overlap=150,
    )

    print_chunk_preview(chunk_records)


if __name__ == "__main__":
    main()