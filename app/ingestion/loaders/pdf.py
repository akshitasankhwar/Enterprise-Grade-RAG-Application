"""PDF Document Loader - Extract text from PDF files"""

from pypdf import PdfReader
from typing import List
from pathlib import Path


def load_pdf(file_path: str) -> List[str]:
    """
    Load a PDF file and extract text from all pages.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        List of page texts
    """
    try:
        reader = PdfReader(file_path)
        pages = []
        
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text.strip():  # Only add non-empty pages
                pages.append(text)
        
        print(f"✅ Extracted {len(pages)} pages from {Path(file_path).name}")
        return pages
    
    except Exception as e:
        print(f"❌ Error loading PDF {file_path}: {e}")
        return []
