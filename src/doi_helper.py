# doi_helper.py - DOI lookup and paper download functions

import os
import time
import requests
import logging
from bs4 import BeautifulSoup
import re
import random
from pathlib import Path
import PyPDF2
import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.71 Safari/537.36",
]

def get_timestamp_str():
    """Get current time as string in YYYY-MM-DD HH:MM:SS format."""
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def get_headers():
    """Return random user agent headers to avoid blocking."""
    return {'User-Agent': random.choice(USER_AGENTS)}

def extract_title(html):
    """Extract title from Sci-Hub HTML page."""
    soup = BeautifulSoup(html, "html.parser")
    title_tag = soup.find("title")
    if title_tag:
        title = title_tag.text.strip()
        title = re.sub(r'^(Sci-Hub\s*-*\s*)', "", title, flags=re.IGNORECASE)
        title = ' '.join(title.split()[:15])
        title = re.sub(r'[\\/*?:"<>|]', "", title)
        return title
    return None

def extract_scihub_embed_link(html):
    """Extract PDF embed link from Sci-Hub HTML page."""
    soup = BeautifulSoup(html, "html.parser")
    embed_tag = soup.find("embed", src=True)
    if embed_tag:
        link = embed_tag.get("src")
        if link.startswith('/downloads'):
            link = "https://sci-hub.se" + link
        return link
    return None

def is_valid_pdf(file_path):
    """Check if PDF file is valid and can be opened."""
    try:
        with open(file_path, 'rb') as f:
            PyPDF2.PdfReader(f)
        return True
    except Exception as e:
        logger.error(f"Invalid PDF file {file_path}: {e}")
        return False

def generate_filename(title, doi):
    """Generate a consistent filename format: title_doi.pdf."""
    # Clean title for filename
    if title:
        clean_title = re.sub(r'[\\/*?:"<>|]', "", title)
        clean_title = clean_title.strip().replace(" ", "_")[:100]  # Limit length
    else:
        clean_title = "unknown_paper"
    
    # Add DOI if available
    if doi:
        clean_doi = doi.replace("/", "_")
        return f"{clean_title}_{clean_doi}.pdf"
    else:
        return f"{clean_title}.pdf"

def download_file(url, title, doi, download_folder, max_retries=3):
    """Download a file from URL with retries and validation."""
    filename = generate_filename(title, doi)
    file_path = os.path.join(download_folder, filename)
    
    # Check if file already exists and is valid
    if os.path.exists(file_path):
        if is_valid_pdf(file_path):
            logger.info(f"Valid file already exists: {file_path}")
            return {"file_path": file_path, "success": True, "status": "skipped", "doi": doi, "timestamp": get_timestamp_str()}
        else:
            logger.warning(f"Found invalid PDF, will re-download: {file_path}")
            os.remove(file_path)
    
    # Retry logic
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=get_headers(), stream=True)
            if response.status_code == 200:
                # Save the file
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                # Validate downloaded PDF
                if is_valid_pdf(file_path):
                    logger.info(f"File successfully downloaded and validated: {file_path}")
                    return {"file_path": file_path, "success": True, "status": "downloaded", "doi": doi, "timestamp": get_timestamp_str()}
                else:
                    logger.warning(f"Downloaded invalid PDF on attempt {attempt+1}, retrying...")
                    os.remove(file_path)
            else:
                logger.error(f"Download failed with status {response.status_code} on attempt {attempt+1}")
        except Exception as e:
            logger.error(f"Error downloading file on attempt {attempt+1}: {e}")
        
        # Wait before retry
        if attempt < max_retries - 1:
            time.sleep(2 * (attempt + 1))  # Progressive backoff
    
    return {"success": False, "status": "download_failed_after_retries", "doi": doi, "timestamp": get_timestamp_str()}

def get_doi_by_title(title):
    """Retrieve DOI using CrossRef API based on paper title."""
    try:
        url = "https://api.crossref.org/works"
        params = {
            "query": title,
            "rows": 1,
            "sort": "score",
        }
        response = requests.get(url, params=params, headers=get_headers())
        if response.status_code == 200:
            data = response.json()
            items = data.get("message", {}).get("items", [])
            if items:
                doi = items[0].get("DOI")
                if doi:
                    logger.info(f"Found DOI for title '{title}': {doi}")
                    return doi
        logger.warning(f"No DOI found for title: {title}")
        return None
    except Exception as e:
        logger.error(f"Error getting DOI for title '{title}': {e}")
        return None

def download(identifier, output_directory="downloads"):
    """Download a paper using DOI or title."""
    os.makedirs(output_directory, exist_ok=True)
    
    # Determine if input is DOI or title
    if isinstance(identifier, str) and identifier.startswith("10."):
        doi = identifier
        title = None
    else:
        title = identifier
        doi = get_doi_by_title(title)
        if not doi:
            return {
                "success": False, 
                "status": "no_doi_found",
                "query": identifier,
                "timestamp": get_timestamp_str()
            }
    
    # Try to download from Sci-Hub
    try:
        base_url = "https://sci-hub.se/"
        sci_hub_url = base_url + doi
        response = requests.get(sci_hub_url, headers=get_headers())
        
        if response.status_code == 200:
            html_content = response.text
            paper_title = extract_title(html_content) or title or doi
            original_url = extract_scihub_embed_link(html_content)
            
            # Fix for malformed URLs starting with '//'
            if original_url and original_url.startswith("//"):
                original_url = "https:" + original_url
                
            if original_url:
                result = download_file(original_url, paper_title, doi, output_directory)
                result["title"] = paper_title
                result["doi"] = doi
                return result
            else:
                return {
                    "success": False, 
                    "status": "no_download_link", 
                    "doi": doi,
                    "title": paper_title,
                    "timestamp": get_timestamp_str()
                }
        else:
            return {
                "success": False, 
                "status": f"sci_hub_error_{response.status_code}",
                "doi": doi,
                "timestamp": get_timestamp_str()
            }
    except Exception as e:
        return {
            "success": False,
            "status": f"error: {str(e)}",
            "doi": doi,
            "timestamp": get_timestamp_str()
        }
