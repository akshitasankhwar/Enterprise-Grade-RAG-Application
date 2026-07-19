"""Text Splitter - Break documents into chunks for embedding"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List


def create_splitter(chunk_size: int = 500, chunk_overlap: int = 50):
    """
    Create a text splitter for chunking documents.
    
    Args:
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks (for context)
        
    Returns:
        RecursiveCharacterTextSplitter instance
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]  # Try to split on natural boundaries
    )


def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """
    Split text into chunks.
    
    Args:
        text: The text to split
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    splitter = create_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    return chunks


if __name__ == "__main__":
    # Test the splitter
    sample_text = "This is a sample text. " * 100  # Repeat to make it long
    chunks = split_text(sample_text)
    print(f"Split into {len(chunks)} chunks")
    print(f"First chunk: {chunks[0][:100]}...")