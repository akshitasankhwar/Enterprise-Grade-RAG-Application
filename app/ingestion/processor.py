"""Document Processor - Ingest all files and store in Qdrant"""

from pathlib import Path
from typing import List, Dict
from app.ingestion.loaders import load_pdf, load_html, load_docx, load_txt
from app.ingestion.chunking import split_text
from app.services.retrieval.embedding import EmbeddingService
from app.services.retrieval.qdrant_service import QdrantService


class DocumentProcessor:
    """
    End-to-end pipeline: Load → Chunk → Embed → Store
    """
    
    def __init__(self):
        """Initialize processor with services."""
        print("\n🚀 Initializing Document Processor...")
        self.embedding_service = EmbeddingService()
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
        elif extension == ".html":
            return load_html(str(file_path))
        elif extension == ".docx":
            return load_docx(str(file_path))
        elif extension in [".txt"]:
            return load_txt(str(file_path))
        else:
            print(f"⚠️ Unsupported file type: {extension}")
            return []
    
    def process_directory(self, directory: str, wipe: bool = False):
        """
        Process all files in a directory recursively.
        
        Args:
            directory: Path to directory containing files
            wipe: If True, recreate Qdrant collection (delete old data)
        """
        print(f"\n📂 Processing directory: {directory}")
        
        if wipe:
            print("🗑️ Wiping existing collection...")
            self.qdrant_service.create_collection(vector_size=384, force_recreate=True)
        
        data_dir = Path(directory)
        all_chunks = []
        total_files = 0
        
        # Process all supported file types
        for file_path in data_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in [".pdf", ".html", ".docx", ".txt"]:
                total_files += 1
                print(f"\n📄 Processing: {file_path.name}")
                
                # Load file
                pages = self.load_file(str(file_path))
                
                # Chunk each page
                for page_idx, page_text in enumerate(pages):
                    chunks = split_text(page_text)
                    
                    for chunk_idx, chunk in enumerate(chunks):
                        all_chunks.append({
                            "text": chunk,
                            "source": str(file_path),
                            "page": page_idx,
                            "chunk_index": chunk_idx
                        })
        
        print(f"\n✅ Loaded {total_files} files, creating {len(all_chunks)} chunks")
        
        # Embed and store
        if all_chunks:
            print(f"🔤 Creating embeddings for {len(all_chunks)} chunks...")
            texts = [chunk["text"] for chunk in all_chunks]
            embeddings = self.embedding_service.embed_texts(texts)
            
            print(f"💾 Storing in Qdrant...")
            self.qdrant_service.add_documents(all_chunks, embeddings)
            
            info = self.qdrant_service.get_collection_info()
            print(f"\n✅ Ingestion complete!")
            print(f"   Files: {total_files}")
            print(f"   Chunks: {len(all_chunks)}")
            print(f"   Collection: {info}")
        else:
            print("⚠️ No documents found!")


def main():
    """Main entry point"""
    import sys
    
    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage: python -m app.ingestion.processor <data_directory> [--wipe]")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    wipe = "--wipe" in sys.argv
    
    # Process
    processor = DocumentProcessor()
    processor.process_directory(data_dir, wipe=wipe)


if __name__ == "__main__":
    main()