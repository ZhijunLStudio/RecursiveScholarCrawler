# src/doi_service.py - DOI查询服务

import requests
import time
import re
from difflib import SequenceMatcher
import logging

logger = logging.getLogger(__name__)

def normalize_title(title):
    """Normalize paper title to improve matching"""
    # Remove special characters, keep letters, numbers and spaces
    clean_title = re.sub(r'[^\w\s]', ' ', title)
    # Convert to lowercase and remove extra spaces
    clean_title = ' '.join(clean_title.lower().split())
    return clean_title

def similarity_score(a, b):
    """Calculate similarity between two strings"""
    return SequenceMatcher(None, normalize_title(a), normalize_title(b)).ratio()

def get_doi_from_title(title, api="crossref"):
    """Query DOI from paper title using API"""
    logger.info(f"Querying DOI for: {title}")
    
    # Set request headers
    headers = {
        'User-Agent': 'DOIFinder/1.0 (mailto:your-email@example.com)',
        'Accept': 'application/json'
    }
    
    if api == "crossref":
        url = "https://api.crossref.org/works"
        params = {
            "query": title,
            "rows": 5,
            "sort": "score"
        }
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                items = data.get("message", {}).get("items", [])
                
                # If results found
                if items:
                    # Find best match
                    best_match = None
                    best_score = 0
                    
                    for item in items:
                        item_title = item.get("title", [""])[0] if item.get("title") else ""
                        if not item_title:
                            continue
                            
                        score = similarity_score(title, item_title)
                        
                        # Only consider results above threshold
                        if score > best_score and score > 0.6:
                            best_score = score
                            best_match = {
                                "doi": item.get("DOI", ""),
                                "title": item_title,
                                "score": score,
                                "publisher": item.get("publisher", ""),
                                "type": item.get("type", ""),
                                "container-title": item.get("container-title", [""])[0] if item.get("container-title") else ""
                            }
                    
                    if best_match:
                        logger.info(f"Found DOI: {best_match['doi']} (score: {best_score:.2f})")
                        return best_match
            
            # If nothing found or API error
            logger.warning(f"Could not find DOI match (status: {response.status_code})")
            return {"doi": None, "error": f"No matching DOI found (status: {response.status_code})"}
                
        except Exception as e:
            logger.error(f"DOI lookup error: {e}")
            return {"doi": None, "error": str(e)}
    
    return {"doi": None, "error": "Unsupported API"}
