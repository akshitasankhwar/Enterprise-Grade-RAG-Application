"""Office Document Loader - Extract text from DOCX files"""

from docx import Document
from typing import List
from pathlib import Path


def load_docx(file_path: str) -> List[str]:
    """
    Load a DOCX (Word) file and extract text from all paragraphs.
    
    Args:
        file_path: Path to the DOCX file
        
    Returns:
        List of paragraph texts
    """
    try:
        doc = Document(file_path)
        pages = []
        
        for para in doc.paragraphs:
            if para.text.strip():  # Only add non-empty paragraphs
                pages.append(para.text)
        
        print(f"✅ Extracted {len(pages)} paragraphs from {Path(file_path).name}")
        return pages
    
    except Exception as e:
        print(f"❌ Error loading DOCX {file_path}: {e}")
        return []
