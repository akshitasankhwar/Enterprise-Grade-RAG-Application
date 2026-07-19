"""Document Loaders - Load various file formats"""

from .pdf import load_pdf
from .html import load_html
from .office import load_docx
from .text import load_txt

__all__ = ["load_pdf", "load_html", "load_docx", "load_txt"]