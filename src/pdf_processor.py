# pdf_processor.py - PDF text extraction and processing functions

import fitz  # PyMuPDF
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {e}")
        return ""

def extract_text_with_limit(pdf_path, max_chars=100000):
    """Extract text from PDF with character limit."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
            if len(text) > max_chars:
                logger.info(f"Reached character limit ({max_chars}), truncating text from {pdf_path}")
                return text[:max_chars] + "... [TRUNCATED]"
        return text
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {e}")
        return ""

def extract_text_by_ratio(pdf_path, head_ratio=None, tail_ratio=None, max_chars=100000):
    """
    Extract text from PDF with specified head and tail ratios.
    
    Args:
        pdf_path: Path to PDF file
        head_ratio: Percentage of text to extract from beginning (0-100)
        tail_ratio: Percentage of text to extract from end (0-100)
        max_chars: Maximum characters to extract
        
    Returns:
        Extracted text according to specified ratios
    """
    try:
        doc = fitz.open(pdf_path)
        all_text = ""
        
        # First pass: extract all text to determine total length
        for page in doc:
            all_text += page.get_text()
        
        total_length = len(all_text)
        
        # If no ratio specified, return full text (with limit)
        if head_ratio is None or tail_ratio is None:
            return all_text[:max_chars] + ("..." if len(all_text) > max_chars else "")
        
        # Calculate character positions based on ratios
        head_chars = int((head_ratio / 100) * total_length)
        tail_chars = int((tail_ratio / 100) * total_length)
        
        # Ensure we don't exceed the total length
        if head_chars + tail_chars > total_length:
            head_chars = min(head_chars, total_length // 2)
            tail_chars = min(tail_chars, total_length - head_chars)
        
        # Extract head and tail portions
        head_text = all_text[:head_chars]
        tail_text = all_text[-tail_chars:] if tail_chars > 0 else ""
        
        # Combine with a separator
        combined_text = head_text
        if head_chars > 0 and tail_chars > 0:
            combined_text += "\n\n[...MIDDLE SECTION OMITTED...]\n\n"
        combined_text += tail_text
        
        # Apply character limit
        if len(combined_text) > max_chars:
            combined_text = combined_text[:max_chars] + "..."
            
        return combined_text
        
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {e}")
        return ""

def get_pdf_info(pdf_path):
    """Get basic information from PDF metadata."""
    try:
        doc = fitz.open(pdf_path)
        info = doc.metadata
        page_count = len(doc)
        return {
            "title": info.get("title", ""),
            "author": info.get("author", ""),
            "subject": info.get("subject", ""),
            "keywords": info.get("keywords", ""),
            "page_count": page_count,
            "path": str(pdf_path)
        }
    except Exception as e:
        logger.error(f"Error getting PDF info from {pdf_path}: {e}")
        return {
            "path": str(pdf_path),
            "error": str(e)
        }
