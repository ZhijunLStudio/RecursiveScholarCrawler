# src/doi_service.py - DOI查询服务 (Final Optimized Version)

import requests
import time
import re
from difflib import SequenceMatcher
import logging

logger = logging.getLogger(__name__)

def normalize_title(title):
    """Normalize paper title to improve matching"""
    clean_title = re.sub(r'[^\w\s]', ' ', title)
    clean_title = ' '.join(clean_title.lower().split())
    return clean_title

def similarity_score(a, b):
    """Calculate similarity between two strings"""
    return SequenceMatcher(None, normalize_title(a), normalize_title(b)).ratio()

def get_doi_from_title(title, api="crossref"):
    """Query DOI from paper title using API"""
    logger.info(f"Querying DOI for: {title}")
    
    headers = {
        'User-Agent': 'RecursiveScholarCrawler/1.1 (mailto:your-email@example.com; https://github.com/your-repo)',
        'Accept': 'application/json'
    }
    
    if api == "crossref":
        # OPTIMIZED: Using 'query.bibliographic' for potentially better title matching,
        # but retaining 'query' as a fallback if needed. The core logic remains our robust one.
        url = "https://api.crossref.org/works"
        params = {
            "query.bibliographic": title, # More specific for bibliographic queries
            "rows": 5,                     # Get a few results to compare
            "sort": "score",
            "order": "desc"
        }
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=15)
            response.raise_for_status()  # Will raise an HTTPError for bad responses (4xx or 5xx)

            data = response.json()
            items = data.get("message", {}).get("items", [])
            
            if not items:
                logger.warning(f"No results from CrossRef for title: {title}")
                return {"doi": None, "error": "No results found"}

            best_match = None
            best_score = 0
            
            for item in items:
                item_title_list = item.get("title")
                if not item_title_list:
                    continue
                
                item_title = item_title_list[0]
                score = similarity_score(title, item_title)
                
                # We want a high similarity score to be confident.
                if score > best_score and score > 0.8: # Using a stricter threshold of 0.8
                    best_score = score
                    best_match = {
                        "doi": item.get("DOI", ""),
                        "title": item_title,
                        "score": score,
                        "publisher": item.get("publisher", ""),
                        "type": item.get("type", ""),
                        "container-title": (item.get("container-title") or [""])[0]
                    }
            
            if best_match:
                logger.info(f"Found best DOI match: {best_match['doi']} (score: {best_score:.2f})")
                return best_match
            else:
                logger.warning(f"Could not find a high-confidence DOI match for title '{title}'. Best score was low.")
                return {"doi": None, "error": "No high-confidence match found"}
                
        except requests.exceptions.RequestException as e:
            logger.error(f"DOI lookup network error: {e}")
            return {"doi": None, "error": str(e)}
        except Exception as e:
            logger.error(f"An unexpected error occurred during DOI lookup: {e}")
            return {"doi": None, "error": str(e)}
    
    return {"doi": None, "error": "Unsupported API"}