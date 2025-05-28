# utils.py - Shared utility functions

import json
import logging
import datetime
import threading
import os
from pathlib import Path
import fitz  # PyMuPDF

class Locker:
    """A simple file locking mechanism using a context manager"""
    def __init__(self, lock_file):
        self.lock_file = lock_file
        self.lock = threading.Lock()
        
    def __enter__(self):
        self.lock.acquire()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.lock.release()
        return False  # Don't suppress exceptions

def get_timestamp_str():
    """Get current time as string in YYYY-MM-DD HH:MM:SS format."""
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def load_json(file_path, default=None):
    """Load JSON file with proper error handling."""
    if not os.path.exists(file_path):
        return default if default is not None else {}
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        logging.warning(f"Corrupted JSON file {file_path}. Using default.")
        return default if default is not None else {}

def save_json(data, file_path):
    """Save data to JSON file with pretty printing."""
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def extract_text_with_limit(pdf_path, max_chars=100000):
    """Extract text from PDF with character limit."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
            if len(text) > max_chars:
                return text[:max_chars] + "... [TRUNCATED]"
        return text
    except Exception as e:
        logging.error(f"Error extracting text from {pdf_path}: {e}")
        return ""

def extract_text_by_ratio(pdf_path, head_ratio=None, tail_ratio=None, max_chars=100000):
    """
    Extract text from PDF with specified head and tail ratios.
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
        logging.error(f"Error extracting text from {pdf_path}: {e}")
        return ""
