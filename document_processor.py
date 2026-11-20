
from pathlib import Path

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def load_document(file_path: str):
    """Load PDF, TXT, or Markdown files without community loaders."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {extension}")

    documents = []

    if extension == ".pdf":
        reader = PdfReader(str(path))

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": path.name,
                        "file_type": "pdf",
                        "page": page_number,
                    },
                )
            )

    else:
        text = path.read_text(encoding="utf-8")

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": path.name,
                    "file_type": extension.lstrip("."),
                },
            )
        )

    return documents


def split_documents(
    documents,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
):
    """Create overlapping text chunks while preserving metadata."""

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index
        chunk.metadata["chunk_size"] = len(chunk.page_content)

    return chunks


def process_document(file_path: str):
    documents = load_document(file_path)
    return split_documents(documents)


if __name__ == "__main__":
    sample_file = Path("documents") / "ai_automation_guide.md"

    chunks = process_document(str(sample_file))

    print(f"\nTotal chunks created: {len(chunks)}")

    for chunk in chunks:
        print("\n" + "=" * 60)
        print("Metadata:", chunk.metadata)
        print("Content:")
        print(chunk.page_content)
