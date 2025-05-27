# paper_crawler.py - Main paper crawler implementation

import logging
import time
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, Any
import queue
import datetime

from .pdf_processor import extract_text_with_limit, extract_text_by_ratio
from .llm_service import LLMService
from . import doi_helper
from .paper_state import PaperState

logger = logging.getLogger(__name__)

class PaperCrawler:
    def __init__(self, 
                 input_dir: str, 
                 output_dir: str,
                 api_base: str,
                 api_key: str,
                 model: str,
                 max_recursion_depth: int = -1,
                 max_workers: int = 5,
                 temperature: Optional[float] = None,
                 top_p: Optional[float] = None,
                 top_k: Optional[int] = None,
                 presence_penalty: Optional[float] = None,
                 max_tokens: Optional[int] = None,
                 text_ratio: Optional[Tuple[int, int]] = None):
        """Initialize the paper crawler with configuration options."""
        
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_recursion_depth = max_recursion_depth
        self.max_workers = max_workers
        self.text_ratio = text_ratio  # Text extraction ratio (head%, tail%)
        
        # Initialize services
        self.llm_service = LLMService(
            api_key=api_key,
            api_base=api_base,
            model=model,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            max_tokens=max_tokens
        )
        
        # Initialize state
        self.state = PaperState(output_dir)
        
        # Process queue and lock
        self.queue = queue.Queue()
        self.lock = threading.Lock()
    
    def _get_timestamp_str(self):
        """Get current time as string in YYYY-MM-DD HH:MM:SS format."""
        return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    def download_paper(self, paper_info: Dict) -> Dict:
        """Download a paper using doi_helper based on reference info."""
        output_path = self.output_dir / "downloads"
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get the title which is mandatory
        title = paper_info.get("ref_title")
        if not title:
            return {"success": False, "status": "missing_title", "timestamp": self._get_timestamp_str()}
        
        try:
            # Use doi_helper to handle DOI finding and downloading
            result = doi_helper.download(title, output_directory=str(output_path))
            
            # Update statistics
            if result.get("success"):
                self.state.increment_stats("download_success_count")
            else:
                self.state.increment_stats("download_fail_count")
            
            return result
        except Exception as e:
            logger.error(f"Error downloading paper '{title}': {e}")
            self.state.increment_stats("download_fail_count")
            return {"success": False, "status": f"error: {str(e)}", "timestamp": self._get_timestamp_str()}
    
    def process_paper(self, pdf_path: Path, current_depth: int = 0) -> Dict:
        """Process a single paper and return its metadata."""
        paper_id = pdf_path.name
        
        # Check if already processed
        if self.state.paper_exists(paper_id):
            logger.info(f"Paper already processed: {paper_id}")
            return self.state.get_paper_details(paper_id)
        
        logger.info(f"[Depth {current_depth}] Processing paper: {paper_id}")
        
        # Extract text from PDF based on ratio if specified
        if self.text_ratio:
            head_ratio, tail_ratio = self.text_ratio
            paper_text = extract_text_by_ratio(pdf_path, head_ratio, tail_ratio, max_chars=100000)
            logger.info(f"[Depth {current_depth}] Extracted text using ratio {head_ratio}%,{tail_ratio}% from {paper_id}")
        else:
            paper_text = extract_text_with_limit(pdf_path, max_chars=100000)
        
        if not paper_text:
            logger.error(f"[Depth {current_depth}] Failed to extract text from {paper_id}")
            paper_info = {
                "path": str(pdf_path),
                "error": "Text extraction failed",
                "processing_depth": current_depth,
                "timestamp": self._get_timestamp_str()
            }
            self.state.update_paper_details(paper_id, paper_info)
            return paper_info
        
        # Query LLM to extract information (now includes usage metrics)
        paper_info = self.llm_service.extract_paper_info(paper_text)
        if not paper_info or (paper_info.get("llm_usage", {}).get("success") is False and len(paper_info) == 1):
            logger.error(f"[Depth {current_depth}] Failed to extract information from {paper_id}")
            paper_info = {
                "path": str(pdf_path),
                "error": "LLM extraction failed",
                "processing_depth": current_depth,
                "timestamp": self._get_timestamp_str(),
                "llm_usage": paper_info.get("llm_usage", {"error": "Unknown LLM error"})
            }
            self.state.update_paper_details(paper_id, paper_info)
            return paper_info
        
        # Add source information
        paper_info["source_pdf"] = str(pdf_path)
        paper_info["processing_depth"] = current_depth
        paper_info["timestamp"] = self._get_timestamp_str()
        
        # Update paper details right away
        self.state.update_paper_details(paper_id, paper_info)
        self.state.increment_stats("processed_count")
        
        # Check if we should continue recursion
        should_recurse = (self.max_recursion_depth == -1) or (current_depth < self.max_recursion_depth)
        
        # Process references
        if "references" in paper_info and should_recurse:
            reference_results = []
            ref_count = len(paper_info["references"])
            self.state.increment_stats("reference_count", ref_count)
            
            logger.info(f"[Depth {current_depth}] Found {ref_count} references in {paper_id}")
            
            for idx, ref in enumerate(paper_info["references"]):
                # Download reference paper using title
                logger.info(f"[Depth {current_depth}] Processing reference {idx+1}/{ref_count}: {ref.get('ref_title', 'Unknown')}")
                download_result = self.download_paper(ref)
                
                # Add download results to reference info
                ref_result = {
                    **ref,  # Original reference info
                    "download_path": download_result.get("file_path", ""),
                    "download_success": download_result.get("success", False),
                    "download_status": download_result.get("status", "unknown"),
                    "download_timestamp": download_result.get("timestamp", self._get_timestamp_str())
                }
                
                # Add found DOI if available
                if "doi" in download_result:
                    ref_result["found_doi"] = download_result["doi"]
                
                reference_results.append(ref_result)
                
                # Update paper with current reference results in real-time
                paper_info["references"] = reference_results
                self.state.update_paper_details(paper_id, paper_info)
                
                # Add to processing queue if downloaded successfully
                if download_result.get("success") and download_result.get("file_path"):
                    self.queue.put((Path(download_result["file_path"]), current_depth + 1))
            
            # Final update with all reference results
            paper_info["references"] = reference_results
            self.state.update_paper_details(paper_id, paper_info)
            
        return paper_info
    
    def worker(self):
        """Worker thread to process papers from queue."""
        while True:
            try:
                pdf_path, depth = self.queue.get(block=False)
                self.process_paper(pdf_path, depth)
                self.queue.task_done()
            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"Error in worker thread: {e}")
                self.queue.task_done()
    
    def crawl(self):
        """Start the crawling process."""
        # First, enumerate all PDFs in the input directory
        pdf_files = list(self.input_dir.glob("**/*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        # Update stats
        self.state.update_stats({
            "total_initial_papers": len(pdf_files),
            "start_time": self._get_timestamp_str()
        })
        
        # Add all PDFs to the queue at depth 0
        for pdf_path in pdf_files:
            self.queue.put((pdf_path, 0))
        
        # Start worker threads
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            while not self.queue.empty():
                futures = []
                
                # Create workers up to max_workers
                for _ in range(min(self.max_workers, self.queue.qsize())):
                    futures.append(executor.submit(self.worker))
                
                # Wait for this batch to complete
                for future in futures:
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Worker thread failed: {e}")
                
                # Update completion status
                self.state.update_stats({
                    "queue_remaining": self.queue.qsize(),
                    "last_batch_time": self._get_timestamp_str()
                })
        
        # Final stats update
        self.state.update_stats({
            "end_time": self._get_timestamp_str(),
            "status": "completed"
        })
        
        logger.info("Crawling completed!")
        
        # Print summary
        stats = self.state.get_stats()
        logger.info(f"Summary: Processed {stats.get('processed_count', 0)} papers, "
                   f"Found {stats.get('reference_count', 0)} references, "
                   f"Downloaded {stats.get('download_success_count', 0)} papers successfully, "
                   f"Failed to download {stats.get('download_fail_count', 0)} papers.")
