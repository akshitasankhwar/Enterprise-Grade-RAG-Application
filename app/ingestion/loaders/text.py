"""Text Document Loader - Load plain text files"""

from typing import List
from pathlib import Path


def load_txt(file_path: str) -> List[str]:
    """
    Load a text file.
    
    Args:
        file_path: Path to the text file
        
    Returns:
        List containing the file content
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if content.strip():
            print(f"✅ Loaded {Path(file_path).name}")
            return [content]
        else:
            print(f"⚠️ File is empty: {file_path}")
            return []
    
    except Exception as e:
        print(f"❌ Error loading TXT {file_path}: {e}")
        return []
