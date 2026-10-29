"""Document Processor - Ingest uploaded files and store them in Qdrant"""

from pathlib import Path
from typing import List, Dict
from app.ingestion.loaders import load_pdf, load_html, load_docx, load_txt
from app.ingestion.splitter import split_text
from app.services.embedding_service import get_embedding_service
from app.services.qdrant_service import QdrantService


# File types the loaders can read
SUPPORTED_EXTENSIONS = [".pdf", ".html", ".htm", ".docx", ".txt", ".md", ".csv"]


class DocumentProcessor:
    """
    End-to-end pipeline for uploaded files: Load → Chunk → Embed → Store
    """

    def __init__(self):
        """Initialize processor with services."""
        print("\n🚀 Initializing Document Processor...")
        self.embedding_service = get_embedding_service()
        self.qdrant_service = QdrantService()
        self.qdrant_service.create_collection(vector_size=384)

    def load_file(self, file_path: str) -> List[str]:
        """
        Load a file based on its extension.

        Args:
            file_path: Path to the file

        Returns:
            List of text content from the file
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()

        if extension == ".pdf":
            return load_pdf(str(file_path))
        elif extension in [".html", ".htm"]:
            return load_html(str(file_path))
        elif extension == ".docx":
            return load_docx(str(file_path))
        elif extension in [".txt", ".md", ".csv"]:
            return load_txt(str(file_path))
        else:
            print(f"⚠️ Unsupported file type: {extension}")
            return []

    def process_files(self, file_paths: List[str]) -> Dict[str, int]:
        """
        Load, chunk, embed and store uploaded files.

        Chunks are stored with the file name as their source, so re-uploading
        a file replaces its previous chunks instead of duplicating them.

        Args:
            file_paths: Paths of the saved uploads

        Returns:
            Mapping of file name → number of chunks stored
        """
        all_chunks = []
        chunk_counts = {}

        for file_path in file_paths:
            filename = Path(file_path).name
            print(f"\n📄 Processing: {filename}")

            # Load file
            pages = self.load_file(file_path)

            # Chunk each page
            file_chunks = []
            for page_idx, page_text in enumerate(pages):
                chunks = split_text(page_text)

                for chunk_idx, chunk in enumerate(chunks):
                    file_chunks.append({
                        "text": chunk,
                        "source": filename,
                        "page": page_idx,
                        "chunk_index": chunk_idx
                    })

            chunk_counts[filename] = len(file_chunks)
            all_chunks.extend(file_chunks)

        print(f"\n✅ Loaded {len(file_paths)} files, creating {len(all_chunks)} chunks")

        # Embed and store
        if all_chunks:
            # Drop old chunks of these files so re-uploads don't duplicate
            for filename, count in chunk_counts.items():
                if count:
                    self.qdrant_service.delete_by_source(filename)

            print(f"🔤 Creating embeddings for {len(all_chunks)} chunks...")
            texts = [chunk["text"] for chunk in all_chunks]
            embeddings = self.embedding_service.embed_texts(texts)

            print(f"💾 Storing in Qdrant...")
            self.qdrant_service.add_documents(all_chunks, embeddings)

            print(f"\n✅ Ingestion complete!")
            print(f"   Files: {len(file_paths)}")
            print(f"   Chunks: {len(all_chunks)}")
        else:
            print("⚠️ No text found in uploaded files!")

        return chunk_counts
