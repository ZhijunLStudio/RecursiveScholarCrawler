# src/paper_downloader.py - SciHub download functionality (Final Corrected Version)

import requests
import time
import random
import os
import re
import logging
from pathlib import Path
from bs4 import BeautifulSoup
import urllib.parse
import fitz  # PyMuPDF

# 导入我们项目中的DOI查询服务
from src.doi_service import get_doi_from_title

logger = logging.getLogger(__name__)

# 你可以维护这个镜像列表
SCIHUB_MIRRORS = [
    "https://sci-hub.se", "https://sci-hub.st", "https://sci-hub.ru",
    "https://www.sci-hub.ren", "https://www.sci-hub.ee",
    # 可以添加或删除
]

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0",
]

def get_random_user_agent():
    return random.choice(USER_AGENTS)

def get_random_mirrors(exclude=None, count=None):
    available = list(set(SCIHUB_MIRRORS)) # 去重
    if exclude:
        available = [m for m in available if m not in exclude]
    random.shuffle(available)
    if count and count < len(available):
        return available[:count]
    return available

def clean_filename(title, doi):
    if title:
        cleaned = re.sub(r'[\\/*?:"<>|]', "", title).strip()[:100].replace(" ", "_")
    else:
        cleaned = "unknown_paper"
    if doi:
        cleaned_doi = doi.replace("/", "_").replace(".", "-")
        return f"{cleaned}_{cleaned_doi}.pdf"
    return f"{cleaned}.pdf"

def find_pdf_link(html_content, base_url):
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        for tag in soup.find_all(['embed', 'iframe']):
            if tag.get('src'):
                src = tag['src']
                if src.startswith('//'): return f"https:{src}"
                if not src.startswith('http'): return urllib.parse.urljoin(base_url, src)
                return src
        for link in soup.find_all('a', href=True):
            if 'pdf' in link['href'].lower() or (link.get('id') == 'download'):
                href = link['href']
                if href.startswith('//'): return f"https:{href}"
                if not href.startswith('http'): return urllib.parse.urljoin(base_url, href)
                return href
        return None
    except Exception as e:
        logger.error(f"Error parsing HTML to find PDF link: {e}")
        return None

def _validate_and_save_pdf(pdf_response, output_path):
    with open(output_path, 'wb') as f:
        for chunk in pdf_response.iter_content(chunk_size=8192):
            f.write(chunk)
    try:
        with fitz.open(output_path) as doc:
            if doc.page_count > 0:
                logger.info(f"✓ Successfully downloaded and validated paper! ({os.path.getsize(output_path)} bytes)")
                return True
    except Exception as e:
        logger.warning(f"Validation failed: downloaded file is not a valid PDF: {e}.")
        if os.path.exists(output_path):
            os.remove(output_path)
    return False

def _download_by_doi(doi, output_dir, filename_title, delay=3, max_mirrors=5):
    """Internal: The ONLY reliable way to download from Sci-Hub, using a DOI."""
    if not doi: return False, None, "DOI is empty"
    os.makedirs(output_dir, exist_ok=True)
    
    for mirror in get_random_mirrors(count=max_mirrors):
        try:
            url = f"{mirror}/{urllib.parse.quote_plus(doi)}"
            logger.info(f"Trying direct download from {url}...")
            
            with requests.Session() as session:
                session.headers.update({'User-Agent': get_random_user_agent()})
                response = session.get(url, timeout=30)

                if response.status_code != 200:
                    logger.warning(f"Cannot access {mirror}, status: {response.status_code}")
                    time.sleep(1)
                    continue

                pdf_link = find_pdf_link(response.text, mirror)
                if not pdf_link:
                    logger.warning(f"PDF link not found on {mirror} for DOI: {doi}")
                    time.sleep(1)
                    continue

                logger.info(f"Found PDF link: {pdf_link}")
                pdf_response = session.get(pdf_link, timeout=60, stream=True)
                if pdf_response.status_code == 200:
                    filename = clean_filename(filename_title, doi)
                    output_path = os.path.join(output_dir, filename)
                    if _validate_and_save_pdf(pdf_response, output_path):
                        return True, output_path, None # Success!
            
            time.sleep(delay) # Wait before trying next mirror
        except Exception as e:
            logger.error(f"Error during direct download from {mirror}: {e}")
            time.sleep(1)

    return False, None, f"Failed all mirror attempts for DOI {doi}"

def download_paper_with_fallback(doi, title, output_dir, delay=3, max_mirrors=5):
    """
    Downloads a paper. If DOI is provided, uses it. 
    If DOI is missing or fails, it first tries to find a DOI using the title, then downloads.
    """
    # Step 1: Use the provided DOI if it exists.
    if doi:
        logger.info(f"Attempting download with provided DOI: {doi}")
        success, path, error = _download_by_doi(doi, output_dir, title, delay, max_mirrors)
        if success:
            return True, path, None
        logger.warning(f"Provided DOI failed: {error}. Will try to find a new one with the title.")

    # Step 2: If no DOI, or if the provided DOI failed, use the title to find a new DOI.
    if not title:
        return False, None, "No DOI and no title provided. Cannot proceed."
    
    logger.info(f"Searching for a new DOI using title: '{title[:70]}...'")
    doi_result = get_doi_from_title(title) # This is our existing function!
    
    new_doi = doi_result.get("doi")
    if not new_doi:
        error_msg = f"Could not find a valid DOI for title '{title[:70]}...'. Error: {doi_result.get('error')}"
        logger.error(error_msg)
        return False, None, error_msg

    # Avoid re-downloading if the found DOI is the one that just failed.
    if new_doi == doi:
        error_msg = f"Found the same DOI ({new_doi}) that already failed. Aborting."
        logger.warning(error_msg)
        return False, None, error_msg

    # Step 3: Attempt download with the newly found DOI.
    logger.info(f"Found a new DOI: {new_doi}. Attempting download with it.")
    success, path, error = _download_by_doi(new_doi, output_dir, title, delay, max_mirrors)
    if success:
        return True, path, None

    # All attempts failed
    final_error = f"All download attempts failed. Last error with DOI {new_doi}: {error}"
    logger.error(final_error)
    return False, None, final_error


def retry_failed_downloads(failed_queue, output_dir, delay=5, max_retries=3):
    """Retry downloading papers that failed initially using the robust fallback logic."""
    if not failed_queue:
        logger.info("No failed downloads to retry")
        return [], []
    
    logger.info(f"Retrying {len(failed_queue)} failed downloads...")
    still_failed, successful = [], []
    
    for i, item in enumerate(failed_queue):
        doi = item.get('doi')
        title = item.get('title', 'Unknown Title')
        retry_count = item.get('retry_count', 0) + 1
        
        if retry_count > max_retries:
            logger.warning(f"Exceeded max retries for {title}, skipping")
            item['retry_count'] = retry_count
            still_failed.append(item)
            continue
            
        logger.info(f"Retry {retry_count}/{max_retries} for [{i+1}/{len(failed_queue)}]: {title}")
        success, path, error = download_paper_with_fallback(doi, title, output_dir, delay=delay)
        
        item_update = {
            'download_success': success, 'local_path': path, 'error_message': error,
            'retry_count': retry_count, 'retry_time': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        item.update(item_update)

        if success:
            logger.info(f"Successfully downloaded on retry {retry_count}")
            successful.append(item)
        else:
            logger.warning(f"Download failed again on retry {retry_count}")
            still_failed.append(item)
        
        if i < len(failed_queue) - 1:
            time.sleep(delay)
    
    return still_failed, successful