#!/usr/bin/env python3
# parser.py - Paper parsing functionality using LLM with recursive reference downloading

import os
import sys
import argparse
import logging
import time
import signal
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor
import threading
import datetime
import shutil

# Import but don't immediately use configuration
import src.config as config
from src.utils import (load_json, save_json, get_timestamp_str, extract_text_with_limit,
                       extract_text_by_ratio, Locker)
from src.llm_service import LLMService
from src.doi_service import get_doi_from_title
# MODIFIED: Import the new fallback downloader and the updated retry function
from src.paper_downloader import download_paper_with_fallback, retry_failed_downloads

from src.config import get_configured_paths

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PaperParser:
    def __init__(self, input_dir, output_dir, api_base, api_key, model,
                 max_workers=4, max_depth=-1, text_ratio=None, save_interval=30, download_delay=3, **llm_params):
        """Initialize the paper parser."""
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        paths = get_configured_paths()
        self.download_dir = Path(paths["download_dir"])
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        # LLM parameters
        self.api_base = api_base
        self.api_key = api_key
        self.model = model
        self.llm_params = llm_params
        self.text_ratio = text_ratio
        
        # Processing settings
        self.max_workers = max_workers
        self.max_depth = max_depth
        self.save_interval = save_interval
        self.download_delay = download_delay
        
        # Initialize LLM service
        self.llm_service = LLMService(
            api_key=api_key,
            api_base=api_base,
            model=model,
            **llm_params
        )
        
        # Initialize state
        self.load_state()
        self.lock = threading.Lock()
        
        # Set auto-save timer
        self.last_save_time = time.time()
        
        # Set signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Currently processing files
        self.current_processing = {}
        
        # Directory mapping for organization
        self.dir_mapping = {}
        
        # Failed and completed downloads
        self.failed_downloads = []
        self.completed_downloads = []
        
    def signal_handler(self, sig, frame):
        """Handle interrupt signals, save state before exiting"""
        logger.info("Received interrupt signal, saving state before exiting...")
        self.save_state()
        self.save_processing_state()
        sys.exit(0)
        
    def load_state(self):
        """Load processing state from files."""
        paths = config.get_configured_paths()
        
        self.paper_details = load_json(paths["paper_details_file"], {})
        
        self.stats = load_json(paths["stats_file"], {
            "processed_count": 0,
            "found_references": 0,
            "downloaded_papers": 0,
            "download_failures": 0,
            "start_time": get_timestamp_str(),
            "last_update": get_timestamp_str(),
            "status": "initializing",
            "max_depth": self.max_depth
        })
        
        progress_data = load_json(paths["progress_file"], {
            "processed_files": [],
            "failed_files": [],
            "pending_files": [],
            "in_progress_files": [],
            "total_files": 0
        })
        self.progress = {
            "processed_files": set(progress_data["processed_files"]),
            "failed_files": set(progress_data["failed_files"]),
            "pending_files": set(progress_data["pending_files"]),
            "in_progress_files": set(progress_data["in_progress_files"]),
            "total_files": progress_data["total_files"]
        }
        
        self.download_queue = load_json(paths["download_queue_file"], [])
        
        self.download_results = load_json(paths["download_results_file"], [])
        self.failed_downloads = [item for item in self.download_results if item.get('download_success') is False]
        self.completed_downloads = [item for item in self.download_results if item.get('download_success') is True]
        
        self.load_processing_state()
        self._update_pending_downloads()
        
    def _update_pending_downloads(self):
        """Update the list of pending downloads from queue"""
        completed_result_dois = {item.get('doi') for item in self.download_results if item.get('download_success') and item.get('doi')}
        
        self.pending_downloads = [
            item for item in self.download_queue 
            if item.get('doi_lookup_status') == 'success' and item.get('doi') not in completed_result_dois
        ]
        logger.debug(f"Updated pending downloads: {len(self.pending_downloads)} items")
        
    def load_processing_state(self):
        """Load processing state from previous interrupted run"""
        paths = config.get_configured_paths()
        processing_state = load_json(paths["processing_state_file"], {
            "in_progress": {},
            "last_update": get_timestamp_str()
        })
        
        self.current_processing = processing_state.get("in_progress", {})
        
        if self.current_processing:
            logger.info(f"Detected {len(self.current_processing)} unfinished files from previous run")
            for file_path, state in self.current_processing.items():
                if file_path not in self.progress["processed_files"] and file_path not in self.progress["failed_files"]:
                    if file_path not in self.progress["pending_files"]:
                        self.progress["pending_files"].add(file_path)
                    if file_path not in self.progress["in_progress_files"]:
                        self.progress["in_progress_files"].add(file_path)
                    logger.info(f"Re-adding unfinished file to queue: {file_path} (depth: {state.get('depth', 0)})")
    
    def save_processing_state(self):
        """Save current processing state"""
        paths = config.get_configured_paths()
        processing_state = {
            "in_progress": self.current_processing,
            "last_update": get_timestamp_str()
        }
        save_json(processing_state, paths["processing_state_file"])
        
    def save_state(self):
        """Save current state to files."""
        paths = config.get_configured_paths()
        with self.lock:
            serializable_progress = {
                "processed_files": list(self.progress["processed_files"]),
                "failed_files": list(self.progress["failed_files"]),
                "pending_files": list(self.progress["pending_files"]),
                "in_progress_files": list(self.progress["in_progress_files"]),
                "total_files": self.progress["total_files"]
            }
            save_json(self.paper_details, paths["paper_details_file"])
            save_json(self.stats, paths["stats_file"])
            save_json(serializable_progress, paths["progress_file"])
            save_json(self.download_queue, paths["download_queue_file"])
            save_json(self.download_results, paths["download_results_file"])
            self.save_processing_state()
            
    def check_autosave(self):
        """Check if it's time for auto-saving state"""
        current_time = time.time()
        if current_time - self.last_save_time > self.save_interval:
            logger.info(f"Auto-saving state (interval: {self.save_interval}s)")
            self.save_state()
            self.last_save_time = current_time
    
    def update_stats(self, key, increment=1):
        """Update a stats counter."""
        with self.lock:
            self.stats[key] = self.stats.get(key, 0) + increment
            self.stats["last_update"] = get_timestamp_str()
    
    def map_output_directory(self, input_path):
        input_path = Path(input_path)
        if input_path.is_relative_to(self.input_dir):
            rel_path = input_path.relative_to(self.input_dir).parent
            output_subdir = self.output_dir / rel_path
        elif input_path.is_relative_to(self.output_dir):
            output_subdir = input_path.parent
        else:
            logger.warning(f"Input path {input_path} is not in the input or output directory. Using default output directory.")
            output_subdir = self.output_dir
        
        output_subdir.mkdir(parents=True, exist_ok=True)
        return output_subdir
    
    def get_download_dir(self, source_path=None):
        return self.download_dir

    def scan_input_directory(self, subdir=None):
        base_dir = self.input_dir
        if subdir:
            base_dir = base_dir / subdir

        paths = get_configured_paths()
        download_dir = Path(paths["download_dir"])

        pdf_files = [
            p for p in base_dir.glob("**/*.pdf")
            if download_dir not in p.parents
        ]

        for pdf_file in pdf_files:
            rel_path = pdf_file.relative_to(self.input_dir).parent
            self.dir_mapping[str(pdf_file)] = self.output_dir / rel_path

        all_files = set(str(path) for path in pdf_files)
        processed = self.progress["processed_files"]
        failed = self.progress["failed_files"]

        pending = list(all_files - processed - failed)
        in_progress = [path for path in pending if path in self.current_processing]
        if in_progress:
            logger.info(f"Found {len(in_progress)} interrupted files from previous run, will prioritize")

        self.progress["pending_files"] = set(pending)
        self.progress["in_progress_files"] = set(in_progress)
        self.progress["total_files"] = len(all_files)

        logger.info(f"Found {len(pending)} new/pending PDF files out of {len(all_files)} total files")
        return in_progress + [p for p in pending if p not in in_progress]

    def process_paper(self, pdf_path, current_depth=0):
        pdf_path = str(pdf_path)
        paper_id = os.path.basename(pdf_path)
        
        with self.lock:
            self.current_processing[pdf_path] = {
                "start_time": get_timestamp_str(),
                "depth": current_depth,
                "status": "processing",
                "paper_id": paper_id
            }
            if pdf_path not in self.progress["in_progress_files"]:
                self.progress["in_progress_files"].add(pdf_path)
            self.save_processing_state()
        
        if paper_id in self.paper_details and self.paper_details[paper_id].get("processed"):
            logger.info(f"Paper already processed: {paper_id}")
            with self.lock:
                if pdf_path in self.current_processing:
                    del self.current_processing[pdf_path]
                self.progress["in_progress_files"].discard(pdf_path)
                self.save_processing_state()
            return self.paper_details[paper_id]
        
        logger.info(f"[Depth {current_depth}] Processing paper: {paper_id}")
        
        if self.max_depth >= 0 and current_depth > self.max_depth:
            logger.info(f"Reached maximum depth ({self.max_depth}) for {paper_id}, skipping further processing")
            with self.lock:
                if pdf_path in self.current_processing:
                    del self.current_processing[pdf_path]
                self.progress["processed_files"].add(pdf_path)
                self.progress["pending_files"].discard(pdf_path)
                self.progress["in_progress_files"].discard(pdf_path)
            self.save_state()
            return {"depth_limit_reached": True, "path": pdf_path}
        
        try:
            with self.lock:
                self.current_processing[pdf_path]["status"] = "extracting_text"
                self.save_processing_state()
            
            if self.text_ratio:
                head_ratio, tail_ratio = self.text_ratio
                paper_text = extract_text_by_ratio(pdf_path, head_ratio, tail_ratio)
            else:
                paper_text = extract_text_with_limit(pdf_path)
            
            if not paper_text:
                logger.error(f"[Depth {current_depth}] Failed to extract text from {paper_id}")
                with self.lock:
                    self.progress["failed_files"].add(pdf_path)
                    self.progress["pending_files"].discard(pdf_path)
                    self.progress["in_progress_files"].discard(pdf_path)
                    if pdf_path in self.current_processing:
                        del self.current_processing[pdf_path]
                self.save_state()
                return {"error": "Text extraction failed", "path": pdf_path}
            
            with self.lock:
                self.current_processing[pdf_path]["status"] = "querying_llm"
                self.save_processing_state()
            
            self.check_autosave()
            paper_info = self.llm_service.extract_paper_info(paper_text)
            
            if not paper_info or (paper_info.get("llm_usage", {}).get("success") is False and len(paper_info) == 1):
                logger.error(f"[Depth {current_depth}] LLM extraction failed for {paper_id}")
                with self.lock:
                    self.progress["failed_files"].add(pdf_path)
                    self.progress["pending_files"].discard(pdf_path)
                    self.progress["in_progress_files"].discard(pdf_path)
                    if pdf_path in self.current_processing:
                        del self.current_processing[pdf_path]
                self.save_state()
                return {"error": "LLM extraction failed", "path": pdf_path}
            
            with self.lock:
                self.current_processing[pdf_path]["status"] = "processing_references"
                self.save_processing_state()
            
            paper_info.update({
                "source_pdf": pdf_path,
                "processing_timestamp": get_timestamp_str(),
                "processing_depth": current_depth
            })
            
            self.check_autosave()
            
            references = paper_info.get("references", [])
            next_depth = current_depth + 1
            
            for ref in references:
                if ref.get("ref_title"):
                    should_queue = True
                    if self.max_depth >= 0 and next_depth > self.max_depth:
                        ref["depth_limit_reached"] = True
                        should_queue = False
                    
                    if should_queue:
                        with self.lock:
                            title = ref.get("ref_title")
                            already_in_queue = any(item["title"] == title for item in self.download_queue)
                            already_downloaded = any(item["title"] == title and item.get("download_success") for item in self.download_results)
                            if not already_in_queue and not already_downloaded:
                                self.download_queue.append({
                                    "title": title,
                                    "authors": ref.get("ref_authors", []),
                                    "year": ref.get("ref_year", ""),
                                    "venue": ref.get("ref_venue", ""),
                                    "source_paper": paper_id,
                                    "source_pdf": pdf_path,
                                    "source_depth": current_depth,
                                    "target_depth": next_depth,
                                    "queued_at": get_timestamp_str(),
                                    "status": "pending",
                                    "doi_lookup_status": "pending"
                                })
            
            self.update_stats("processed_count")
            self.update_stats("found_references", len(references))
            
            paper_output_dir = self.map_output_directory(pdf_path)
            paper_info_filename = f"{Path(pdf_path).stem}_details.json"
            save_json(paper_info, paper_output_dir / paper_info_filename)
            
            with self.lock:
                self.paper_details[paper_id] = {
                    "processed": True,
                    "output_path": str(paper_output_dir / paper_info_filename),
                    "depth": current_depth
                }
                self.progress["processed_files"].add(pdf_path)
                self.progress["pending_files"].discard(pdf_path)
                self.progress["in_progress_files"].discard(pdf_path)
                if pdf_path in self.current_processing:
                    del self.current_processing[pdf_path]
            
            self.save_state()
            return {"status": "processed", "path": pdf_path, "output_details_path": str(paper_output_dir / paper_info_filename)}
            
        except Exception as e:
            logger.error(f"[Depth {current_depth}] Error processing {paper_id}: {e}", exc_info=True)
            with self.lock:
                self.progress["failed_files"].add(pdf_path)
                self.progress["pending_files"].discard(pdf_path)
                self.progress["in_progress_files"].discard(pdf_path)
                if pdf_path in self.current_processing:
                    del self.current_processing[pdf_path]
            self.save_state()
            return {"error": str(e), "path": pdf_path}
            
    def lookup_doi_for_queue(self):
        processed_dois = {item.get('doi') for item in self.download_results if item.get('download_success')}
        pending_lookups = [
            item for item in self.download_queue
            if item["doi_lookup_status"] == "pending" and item.get("doi") not in processed_dois
        ]

        if not pending_lookups:
            logger.info("No pending DOI lookups.")
            return

        logger.info(f"Looking up DOIs for {len(pending_lookups)} references.")

        for i, item in enumerate(pending_lookups):
            title = item["title"]
            logger.info(f"[{i+1}/{len(pending_lookups)}] Looking up DOI for: {title}")

            if item.get("doi") in processed_dois:
                item["doi_lookup_status"] = "success"
                continue

            max_retries = 3
            initial_delay = 2
            
            for attempt in range(max_retries):
                if attempt > 0:
                    current_delay = initial_delay * (2 ** (attempt - 1))
                    logger.warning(f"Retrying DOI lookup for '{title}' (Attempt {attempt+1}/{max_retries}), waiting {current_delay} seconds...")
                    time.sleep(current_delay)

                try:
                    result = get_doi_from_title(title)
                    item.update({
                        "doi_lookup_result": result,
                        "doi_lookup_time": get_timestamp_str()
                    })

                    if result.get("doi"):
                        item["doi"] = result["doi"]
                        item["doi_lookup_status"] = "success"
                        logger.info(f"✓ Found DOI: {result['doi']}")
                        break
                    else:
                        item["doi_lookup_status"] = "failed"
                        item["doi_lookup_error"] = result.get("error", "Unknown error")
                        if "Connection" not in str(result.get("error", "")):
                            break
                except Exception as e:
                    item["doi_lookup_status"] = "failed"
                    item["doi_lookup_error"] = str(e)
                    logger.error(f"Error during DOI lookup for '{title}': {e}", exc_info=True)
            
            if (i + 1) % 5 == 0:
                self.save_state()
            time.sleep(1) # Pace requests
        
        self._update_pending_downloads()
        self.save_state()
                    
    def download_papers_from_queue(self, batch_size=20, batch_delay=300):
        """Download papers that have DOIs from the download queue in batches"""
        self._update_pending_downloads()
        
        current_batch_to_download = list(self.pending_downloads) 
        
        if not current_batch_to_download:
            logger.info("No papers to download from queue.")
            return
            
        logger.info(f"Starting download for {len(current_batch_to_download)} papers in queue")
        
        for batch_idx in range(0, len(current_batch_to_download), batch_size):
            batch = current_batch_to_download[batch_idx:batch_idx+batch_size]
            logger.info(f"Processing batch {batch_idx//batch_size + 1}/{(len(current_batch_to_download)+batch_size-1)//batch_size} ({len(batch)} papers)")
            
            processed_dois_in_batch = set()

            for i, item in enumerate(batch):
                doi = item.get("doi")
                title = item["title"]
                
                download_dir = self.get_download_dir()
                logger.info(f"[{i+1}/{len(batch)}] Downloading: {title} (DOI: {doi or 'N/A'})")
                item["download_attempt_time"] = get_timestamp_str()
                
                try:
                    # MODIFIED: Use the new fallback download function
                    success, local_path, error = download_paper_with_fallback(
                        doi, title, str(download_dir), delay=self.download_delay
                    )
                    
                    item.update({
                        "download_status": "complete" if success else "failed",
                        "download_success": success,
                        "download_time": get_timestamp_str()
                    })
                    
                    if success and local_path:
                        item["local_path"] = local_path
                        logger.info(f"✓ Successfully downloaded paper to {local_path}")
                        self.update_stats("downloaded_papers")
                        
                        self.completed_downloads.append(item.copy())
                        self.download_results.append(item.copy())
                        
                        if self.max_depth < 0 or item["target_depth"] <= self.max_depth:
                            if not hasattr(self, 'pending_recursive_papers'):
                                self.pending_recursive_papers = []
                            self.pending_recursive_papers.append({
                                "path": local_path,
                                "depth": item["target_depth"]
                            })
                        if doi:
                            processed_dois_in_batch.add(doi)
                    else:
                        logger.warning(f"✗ Failed to download paper: {error}")
                        item["error_message"] = error
                        self.update_stats("download_failures")
                        self.failed_downloads.append(item.copy())
                        self.download_results.append(item.copy())
                        
                except Exception as e:
                    logger.error(f"Critical error during download process for {title}: {e}", exc_info=True)
                    item.update({
                        "download_status": "error",
                        "download_error": str(e)
                    })
                    self.update_stats("download_failures")
                    self.failed_downloads.append(item.copy())
                    self.download_results.append(item.copy())
                    
                if (i + 1) % 2 == 0:
                    self.save_state()
                if i < len(batch) - 1:
                    time.sleep(self.download_delay)
            
            with self.lock:
                self.download_queue = [
                    q_item for q_item in self.download_queue 
                    if not (q_item.get('doi') and q_item.get('doi') in processed_dois_in_batch)
                ]
                self._update_pending_downloads() 
            
            self.save_state()
            
            if batch_idx + batch_size < len(current_batch_to_download):
                logger.info(f"Batch complete, pausing for {batch_delay//60} minutes before next batch...")
                time.sleep(batch_delay)
        
        self.save_state()
    
    def process_references(self, workers=None):
        """Process papers from the download queue up to max depth"""
        if self.max_depth == 0:
            logger.info("Max depth is 0, skipping references processing")
            return
            
        self.pending_recursive_papers = []
        self.lookup_doi_for_queue()
        self.download_papers_from_queue()
        
        if self.failed_downloads:
            logger.info(f"Retrying {len(self.failed_downloads)} failed downloads...")
            download_dir = self.get_download_dir()
            still_failed, newly_successful = retry_failed_downloads(
                self.failed_downloads, str(download_dir), delay=self.download_delay
            )
            self.failed_downloads = still_failed
            for item in newly_successful:
                self.completed_downloads.append(item)
                if self.max_depth < 0 or item["target_depth"] <= self.max_depth:
                    if not hasattr(self, 'pending_recursive_papers'): self.pending_recursive_papers = []
                    self.pending_recursive_papers.append({"path": item["local_path"], "depth": item["target_depth"]})
            self.update_stats("downloaded_papers", len(newly_successful))
            self.update_stats("download_failures", -len(newly_successful))
            self.save_state()
            
        if self.pending_recursive_papers:
            logger.info(f"Processing {len(self.pending_recursive_papers)} downloaded reference papers recursively.")
            papers_to_process = list(self.pending_recursive_papers) 
            self.pending_recursive_papers.clear()

            with ThreadPoolExecutor(max_workers=workers or self.max_workers) as executor:
                futures = [executor.submit(self.process_paper, item["path"], item["depth"]) for item in papers_to_process]
                for future in futures:
                    future.result()
            
            logger.info("Starting recursive reference processing for next depth.")
            self.process_references(workers)
        else:
            logger.info("No more pending recursive papers to process at this depth.")
    
    def process_all_papers(self, subdir=None):
        """Process all pending papers in parallel."""
        pending_files = self.scan_input_directory(subdir)
        
        if pending_files:
            logger.info(f"Starting initial paper processing for {len(pending_files)} files.")
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(self.process_paper, pdf_path, 0) for pdf_path in pending_files]
                for future in futures:
                    future.result()

        logger.info("Initial paper processing complete. Starting recursive reference handling.")
        self.process_references()
        
        self._update_pending_downloads()
        if self.pending_downloads or self.failed_downloads:
            logger.info("Running one more cycle for pending/failed downloads.")
            self.process_references()
        
        if not self.progress["pending_files"] and not self.pending_downloads and not self.failed_downloads and not self.current_processing:
            self.stats["status"] = "completed"
            logger.info("All tasks appear to be completed!")
        else:
            self.stats["status"] = "idle"
            logger.info("Processing finished, but some pending/failed tasks might remain.")
        self.save_state()
            
    def retry_failed_downloads_only(self):
        """Standalone function to retry just the failed downloads"""
        if not self.failed_downloads:
            logger.info("No failed downloads to retry.")
            return
            
        logger.info(f"Retrying {len(self.failed_downloads)} failed downloads...")
        download_dir = self.get_download_dir()
        still_failed, newly_successful = retry_failed_downloads(
            self.failed_downloads, str(download_dir), delay=self.download_delay, max_retries=5
        )
                                             
        newly_successful_count = len(newly_successful)
        if newly_successful_count > 0:
            self.update_stats("downloaded_papers", newly_successful_count)
            self.update_stats("download_failures", -newly_successful_count)
            
            pending_recursive = []
            for item in newly_successful:
                if self.max_depth < 0 or item["target_depth"] <= self.max_depth:
                    pending_recursive.append({"path": item["local_path"], "depth": item["target_depth"]})
            
            for item in self.download_results:
                for success in newly_successful:
                    if item.get("doi") == success.get("doi"):
                        item.update(success)
            
            if pending_recursive:
                logger.info(f"Processing {len(pending_recursive)} newly downloaded papers")
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = [executor.submit(self.process_paper, item["path"], item["depth"]) for item in pending_recursive]
                    for f in futures: f.result()
        
        self.failed_downloads = still_failed
        # Update download_results to reflect remaining failures
        for item in self.download_results:
            for failed_item in still_failed:
                if item.get('doi') == failed_item.get('doi'):
                    item.update(failed_item)

        self.save_state()
        logger.info(f"Retry complete: {newly_successful_count} succeeded, {len(still_failed)} still failed")

def validate_api_url(url):
    """Validate and fix API URL if needed."""
    if not url.startswith('http'):
        return 'https://' + url.lstrip('://')
    return url

def parse_text_ratio(ratio_str):
    """Parse the text extraction ratio parameter."""
    if not ratio_str:
        return None
    try:
        head, tail = map(int, ratio_str.split(','))
        if head < 0 or tail < 0 or head + tail > 100: return None
        return (head, tail)
    except ValueError:
        return None

def main():
    parser = argparse.ArgumentParser(description='Academic Paper Parser using LLM with reference downloading')
    parser.add_argument('--input-dir', required=True, help='Directory containing input PDF files')
    parser.add_argument('--output-dir', required=True, help='Directory to store output files')
    parser.add_argument('--api-base', required=True, help='LLM API base URL')
    parser.add_argument('--api-key', default="EMPTY", help='API key for the LLM service')
    parser.add_argument('--model', required=True, help='LLM model name')
    parser.add_argument('--max-workers', type=int, default=4, help='Maximum number of parallel workers')
    parser.add_argument('--max-depth', type=int, default=-1, help='Maximum recursion depth (-1 for unlimited)')
    parser.add_argument('--text-ratio', default=None, help='Text extraction ratio as "head,tail" (e.g., "10,30")')
    parser.add_argument('--save-interval', type=int, default=30, help='Auto-save interval in seconds')
    parser.add_argument('--download-delay', type=int, default=3, help='Delay between downloads in seconds')
    parser.add_argument('--temperature', type=float, help='LLM temperature parameter')
    parser.add_argument('--top-p', type=float, help='LLM top-p parameter')
    parser.add_argument('--top-k', type=int, help='LLM top-k parameter')
    parser.add_argument('--presence-penalty', type=float, help='LLM presence penalty parameter')
    parser.add_argument('--max-tokens', type=int, help='LLM max tokens parameter')
    parser.add_argument('--subdir', default=None, help='Optional subdirectory to process within input directory')
    # MODIFIED: Renamed for clarity. Only retries and then exits.
    parser.add_argument('--retry-downloads-only', action='store_true', help='Only retry all previously failed downloads and then exit.')
    # NEW: Flag to disable automatic retry on startup
    parser.add_argument('--no-retry-on-start', action='store_true', help='Disable the automatic retry of failed downloads at the beginning of the run.')
    parser.add_argument('--batch-size', type=int, default=20, help='Number of papers to download in each batch')
    parser.add_argument('--batch-delay', type=int, default=300, help='Delay between download batches in seconds')
    
    args = parser.parse_args()
    
    config.configure_paths(args.input_dir, args.output_dir, args.subdir)
    paths = config.get_configured_paths()
    logger.info(f"Configured paths: Input={paths['input_dir']}, Output={paths['output_dir']}")
    
    api_base = validate_api_url(args.api_base)
    text_ratio = parse_text_ratio(args.text_ratio)
    
    llm_params = {k: v for k, v in vars(args).items() if k in ['temperature', 'top_p', 'top_k', 'presence_penalty', 'max_tokens'] and v is not None}
    
    try:
        paper_parser = PaperParser(
            input_dir=args.input_dir, output_dir=args.output_dir, api_base=api_base,
            api_key=args.api_key, model=args.model, max_workers=args.max_workers,
            max_depth=args.max_depth, text_ratio=text_ratio, save_interval=args.save_interval,
            download_delay=args.download_delay, **llm_params
        )
        
        if args.retry_downloads_only:
            logger.info("Running in 'retry failed downloads only' mode.")
            paper_parser.retry_failed_downloads_only()
            logger.info("Retry process finished. Exiting.")
            return

        # NEW: Automatic retry of failed downloads on startup (default behavior)
        if not args.no_retry_on_start:
            if paper_parser.failed_downloads:
                logger.info("--- Starting automatic retry of previously failed downloads ---")
                paper_parser.retry_failed_downloads_only()
                logger.info("--- Automatic retry finished. Continuing with main process. ---")
            else:
                logger.info("No previously failed downloads to retry.")
        else:
            logger.info("Skipping automatic retry of failed downloads as per --no-retry-on-start flag.")

        logger.info(f"Starting main paper processing with max depth: {args.max_depth}...")
        paper_parser.process_all_papers(args.subdir)
        
    except KeyboardInterrupt:
        logger.info("Parser stopped by user")
    except Exception as e:
        logger.error(f"Parser stopped due to a critical error: {e}", exc_info=True)

if __name__ == "__main__":
    main()