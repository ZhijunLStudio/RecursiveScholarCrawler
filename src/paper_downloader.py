# src/paper_downloader.py - SciHub download functionality

import requests
import time
import random
import os
import re
import logging
from pathlib import Path
from bs4 import BeautifulSoup
import urllib.parse

logger = logging.getLogger(__name__)

# 更新为扩展的Sci-Hub镜像列表
SCIHUB_MIRRORS = [
    "https://www.sci-hub.cat",
    "https://www.sci-hub.ren",
    "https://www.sci-hub.st",
    "https://www.sci-hub.se",
    "https://www.sci-hub.ru",
    "https://www.sci-hub.ee",
    "https://www.sci-hub.in", 
    "https://www.sci-hub.vg",
    "https://www.sci-hub.al",
    "https://www.pismin.com",
    "https://www.tesble.com",
    "https://www.wellesu.com",
    "https://sci-hub.usualwant.com",
    "https://www.sci-hub.ru",
    "https://sci-hub.se",
    "https://sci-hub.ee", 
]

# User agent list to avoid bot detection
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.71 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_5_2) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.164 Safari/537.36 Edg/91.0.864.71"
]

def get_random_user_agent():
    """Return a random user agent string"""
    return random.choice(USER_AGENTS)

def get_random_mirrors(exclude=None, count=None):
    """Return a randomized list of mirrors, optionally excluding some"""
    available = SCIHUB_MIRRORS.copy()
    if exclude:
        available = [m for m in available if m not in exclude]
        
    random.shuffle(available)
    if count and count < len(available):
        return available[:count]
    return available

def clean_filename(title, doi):
    """Generate a suitable filename from title and DOI"""
    if title:
        # Clean illegal characters
        cleaned = re.sub(r'[\\/*?:"<>|]', "", title)
        # Limit length
        cleaned = cleaned.strip()[:100]
        # Replace spaces
        cleaned = cleaned.replace(" ", "_")
    else:
        cleaned = "unknown"
    
    # Add DOI if available
    if doi:
        cleaned_doi = doi.replace("/", "_")
        return f"{cleaned}_{cleaned_doi}.pdf"
    else:
        return f"{cleaned}.pdf"

def find_pdf_link(html_content, base_url):
    """Extract PDF download link from SciHub page"""
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Try multiple methods to find PDF link
        
        # 1. Check for embed tag
        embed = soup.find('embed')
        if embed and embed.get('src'):
            src = embed.get('src')
            if src.startswith('//'):
                return f"https:{src}"
            elif src.startswith('/'):
                return f"{base_url}{src}"
            return src
        
        # 2. Check for iframe
        iframe = soup.find('iframe')
        if iframe and iframe.get('src'):
            src = iframe.get('src')
            if src.startswith('//'):
                return f"https:{src}"
            elif src.startswith('/'):
                return f"{base_url}{src}"
            return src
        
        # 3. Look for download button
        download_btn = soup.find('a', id='download')
        if download_btn and download_btn.get('href'):
            href = download_btn.get('href')
            if href.startswith('//'):
                return f"https:{href}"
            elif href.startswith('/'):
                return f"{base_url}{href}"
            return href
        
        # 4. Check for links with PDF keyword
        pdf_links = soup.find_all('a', href=lambda href: href and ('pdf' in href.lower()))
        for link in pdf_links:
            href = link.get('href')
            if href.startswith('//'):
                return f"https:{href}"
            elif href.startswith('/'):
                return f"{base_url}{href}"
            return href
        
        # 5. Find potential download area
        download_div = soup.find('div', id='download')
        if download_div:
            buttons = download_div.find_all('button')
            for button in buttons:
                onclick = button.get('onclick', '')
                match = re.search(r"location.href='(.+?)'", onclick)
                if match:
                    href = match.group(1)
                    if href.startswith('//'):
                        return f"https:{href}"
                    elif href.startswith('/'):
                        return f"{base_url}{href}"
                    return href
        
        # 6. Extract anything that looks like a PDF link
        for link in soup.find_all('a'):
            href = link.get('href', '')
            if href and (href.endswith('.pdf') or '/pdf/' in href):
                if href.startswith('//'):
                    return f"https:{href}"
                elif href.startswith('/'):
                    return f"{base_url}{href}"
                return href
        
        return None
    except Exception as e:
        logger.error(f"Error parsing HTML: {e}")
        return None

def download_from_scihub(doi, output_dir, delay=3, max_mirrors=5):
    """Download paper with given DOI from SciHub using random mirrors"""
    if not doi:
        logger.error("Error: DOI is required")
        return False, None, None
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get random mirrors, limit to max_mirrors to avoid excessive tries
    tried_mirrors = []
    success = False
    output_path = None
    error_message = "Failed with all mirrors"
    
    # 尝试随机选择的镜像，如果失败则继续尝试其他镜像
    while len(tried_mirrors) < min(max_mirrors, len(SCIHUB_MIRRORS)):
        # 随机选择一个未尝试过的镜像
        available_mirrors = get_random_mirrors(exclude=tried_mirrors, count=1)
        if not available_mirrors:
            break
            
        mirror = available_mirrors[0]
        tried_mirrors.append(mirror)
        
        try:
            # Build SciHub URL
            url = f"{mirror}/{doi}"
            logger.info(f"Trying to download from {url}...")
            
            # Set request headers
            headers = {
                'User-Agent': get_random_user_agent(),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            # Get SciHub page
            response = requests.get(url, headers=headers, timeout=30)
            
            if response.status_code != 200:
                logger.warning(f"Cannot access {mirror}, status: {response.status_code}")
                error_message = f"HTTP error {response.status_code} with {mirror}"
                time.sleep(1)  # Brief pause before trying next mirror
                continue
            
            # Extract PDF download link
            pdf_link = find_pdf_link(response.text, mirror)
            
            if not pdf_link:
                logger.warning(f"PDF download link not found on {mirror}")
                error_message = f"No PDF link found on {mirror}"
                time.sleep(1)  # Brief pause before trying next mirror
                continue
            
            logger.info(f"Found PDF link: {pdf_link}")
            
            # Determine filename
            # Try to extract title from page
            soup = BeautifulSoup(response.text, 'html.parser')
            page_title = soup.title.string if soup.title else None
            if page_title and "sci-hub" in page_title.lower():
                # Remove "Sci-Hub | " prefix
                page_title = re.sub(r'^Sci-Hub\s*[|:]\s*', '', page_title, flags=re.IGNORECASE)
            
            filename = clean_filename(page_title, doi)
            output_path = os.path.join(output_dir, filename)
            
            # Download PDF
            logger.info(f"Downloading PDF to: {output_path}")
            pdf_headers = {
                'User-Agent': get_random_user_agent(),
                'Accept': 'application/pdf,application/octet-stream',
                'Referer': url,
            }
            
            pdf_response = requests.get(pdf_link, headers=pdf_headers, timeout=60, stream=True)
            
            if pdf_response.status_code == 200:
                with open(output_path, 'wb') as f:
                    for chunk in pdf_response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                # Check file size
                file_size = os.path.getsize(output_path)
                if file_size < 1000:  # If less than 1KB, might not be valid PDF
                    logger.warning(f"Warning: Downloaded file is very small ({file_size} bytes), may not be valid PDF")
                    # Read file content to check for error messages
                    with open(output_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read(500)
                        logger.warning(f"File content preview: {content}")
                    error_message = f"Downloaded file too small ({file_size} bytes)"
                    os.remove(output_path)  # Remove invalid file
                    time.sleep(1)  # Brief pause before trying next mirror
                    continue
                
                logger.info(f"✓ Successfully downloaded paper! ({file_size} bytes)")
                success = True
                break  # Successfully downloaded, exit the loop
            else:
                logger.warning(f"Failed to download PDF, status: {pdf_response.status_code}")
                error_message = f"PDF download failed with status {pdf_response.status_code}"
                time.sleep(1)  # Brief pause before trying next mirror
        
        except Exception as e:
            logger.error(f"Error downloading from {mirror}: {e}")
            error_message = f"Error with {mirror}: {str(e)}"
            time.sleep(1)  # Brief pause before trying next mirror
        
    # All mirrors tried or success
    if success:
        return True, output_path, None
    else:
        logger.error(f"❌ All mirrors failed: {error_message}")
        return False, None, error_message

def retry_failed_downloads(failed_queue, output_dir, delay=5, max_retries=3):
    """Retry downloading papers that failed initially"""
    if not failed_queue:
        logger.info("No failed downloads to retry")
        return []
    
    logger.info(f"Retrying {len(failed_queue)} failed downloads...")
    still_failed = []
    successful = []
    
    for i, item in enumerate(failed_queue):
        doi = item.get('doi')
        title = item.get('title', 'Unknown Title')
        retry_count = item.get('retry_count', 0) + 1
        
        if retry_count > max_retries:
            logger.warning(f"Exceeded max retries for {title}, skipping")
            item['retry_count'] = retry_count
            still_failed.append(item)
            continue
            
        logger.info(f"Retry {retry_count}/{max_retries} for [{i+1}/{len(failed_queue)}]: {title} (DOI: {doi})")
        
        if doi:
            success, path, error = download_from_scihub(doi, output_dir, delay=delay)
            if success:
                item['download_success'] = True
                item['local_path'] = path
                item['download_status'] = 'complete'
                item['retry_count'] = retry_count
                item['retry_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
                logger.info(f"Successfully downloaded on retry {retry_count}")
                successful.append(item)
            else:
                item['download_success'] = False
                item['download_status'] = 'failed'
                item['retry_count'] = retry_count
                item['retry_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
                item['error_message'] = error
                still_failed.append(item)
                logger.warning(f"Download failed again on retry {retry_count}")
        else:
            logger.warning(f"No DOI available for {title}, cannot download")
            item['error'] = "No DOI available"
            still_failed.append(item)
        
        # Add delay between retries
        if i < len(failed_queue) - 1:
            time.sleep(delay)
    
    return still_failed, successful
