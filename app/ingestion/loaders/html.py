"""HTML Document Loader - Extract text from HTML files"""

from bs4 import BeautifulSoup
from typing import List
from pathlib import Path


def load_html(file_path: str) -> List[str]:
    """
    Load an HTML file and extract clean text.
    
    Args:
        file_path: Path to the HTML file
        
    Returns:
        List containing the extracted text
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text
        text = soup.get_text(separator=' ')
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        print(f"✅ Extracted text from {Path(file_path).name}")
        return [text] if text.strip() else []
    
    except Exception as e:
        print(f"❌ Error loading HTML {file_path}: {e}")
        return []


if __name__ == "__main__":
    # Test the loader
    test_html = "data/true_data/sample.html"
    result = load_html(test_html)
    print(f"Loaded {len(result)} chunk(s)")