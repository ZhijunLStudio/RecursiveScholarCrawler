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
from src.paper_downloader import download_from_scihub, retry_failed_downloads

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
        # Get paths from configuration
        paths = config.get_configured_paths()
        
        # Load paper details
        self.paper_details = load_json(paths["paper_details_file"], {})
        
        # Load or initialize stats
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
        
        # Load or initialize progress tracker
        self.progress = load_json(paths["progress_file"], {
            "processed_files": [],
            "failed_files": [],
            "pending_files": [],
            "in_progress_files": [],
            "total_files": 0
        })
        
        # Initialize download queue
        self.download_queue = load_json(paths["download_queue_file"], [])
        
        # Initialize download results and identify failed downloads for retry
        self.download_results = load_json(paths["download_results_file"], [])
        self.failed_downloads = [item for item in self.download_results if item.get('download_success') is False]
        self.completed_downloads = [item for item in self.download_results if item.get('download_success') is True]
        
        # Load processing state
        self.load_processing_state()
        
        # Calculate pending downloads - items in queue but not in results
        self._update_pending_downloads()
        
    def _update_pending_downloads(self):
        """Update the list of pending downloads from queue"""
        # Find DOIs that are in download queue but not in results
        queue_dois = set([item.get('doi') for item in self.download_queue if item.get('doi')])
        result_dois = set([item.get('doi') for item in self.download_results])
        
        # Identify pending downloads
        self.pending_downloads = [
            item for item in self.download_queue 
            if item.get('doi') and item.get('doi') not in result_dois and item.get('doi_lookup_status') == 'success'
        ]
        
    def load_processing_state(self):
        """Load processing state from previous interrupted run"""
        paths = config.get_configured_paths()
        processing_state = load_json(paths["processing_state_file"], {
            "in_progress": {},
            "last_update": get_timestamp_str()
        })
        
        self.current_processing = processing_state.get("in_progress", {})
        
        # Check for unfinished processing
        if self.current_processing:
            logger.info(f"Detected {len(self.current_processing)} unfinished files from previous run")
            # Add unfinished files back to the processing queue
            for file_path, state in self.current_processing.items():
                if file_path not in self.progress["processed_files"] and file_path not in self.progress["failed_files"]:
                    if file_path not in self.progress["pending_files"]:
                        self.progress["pending_files"].append(file_path)
                    if file_path not in self.progress["in_progress_files"]:
                        self.progress["in_progress_files"].append(file_path)
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
            save_json(self.paper_details, paths["paper_details_file"])
            save_json(self.stats, paths["stats_file"])
            save_json(self.progress, paths["progress_file"])
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
        """Map input path to corresponding output directory structure"""
        input_path = Path(input_path)
        rel_path = input_path.relative_to(self.input_dir).parent
        output_subdir = self.output_dir / rel_path
        output_subdir.mkdir(parents=True, exist_ok=True)
        return output_subdir

    def get_download_dir(self, source_path=None):
        """Get appropriate download directory based on source path"""
        if source_path:
            source_dir = self.map_output_directory(source_path)
            download_dir = source_dir / "downloads"
        else:
            download_dir = self.output_dir / "downloads"
        
        download_dir.mkdir(parents=True, exist_ok=True)
        return download_dir

    def scan_input_directory(self, subdir=None):
        """
        Scan input directory for PDF files and update pending list.
        If subdir is specified, only scan that subdirectory.
        """
        base_dir = self.input_dir
        if subdir:
            base_dir = base_dir / subdir
            
        pdf_files = list(base_dir.glob("**/*.pdf"))
        
        # Map each input file to its corresponding output directory
        for pdf_file in pdf_files:
            rel_path = pdf_file.relative_to(self.input_dir).parent
            self.dir_mapping[str(pdf_file)] = self.output_dir / rel_path
        
        # First pass: get all file paths
        all_files = set(str(path) for path in pdf_files)
        processed = set(self.progress["processed_files"])
        failed = set(self.progress["failed_files"])
        
        # Identify new or pending files
        pending = list(all_files - processed - failed)
        
        # Check for previously interrupted files
        in_progress = [path for path in pending if path in self.current_processing]
        if in_progress:
            logger.info(f"Found {len(in_progress)} interrupted files from previous run, will prioritize")
        
        # Update progress tracker
        self.progress["pending_files"] = pending
        self.progress["in_progress_files"] = in_progress
        self.progress["total_files"] = len(all_files)
        
        logger.info(f"Found {len(pending)} new/pending PDF files out of {len(all_files)} total files")
        return in_progress + [p for p in pending if p not in in_progress]
    
    def process_paper(self, pdf_path, current_depth=0):
        """Process a single paper using LLM and update state."""
        pdf_path = str(pdf_path)
        paper_id = os.path.basename(pdf_path)
        
        # Set current processing file status
        with self.lock:
            self.current_processing[pdf_path] = {
                "start_time": get_timestamp_str(),
                "depth": current_depth,
                "status": "processing",
                "paper_id": paper_id
            }
            
            if pdf_path not in self.progress["in_progress_files"]:
                self.progress["in_progress_files"].append(pdf_path)
                
            self.save_processing_state()
        
        # Check if already processed
        if paper_id in self.paper_details:
            logger.info(f"Paper already processed: {paper_id}")
            
            # Clean up processing state
            with self.lock:
                if pdf_path in self.current_processing:
                    del self.current_processing[pdf_path]
                if pdf_path in self.progress["in_progress_files"]:
                    self.progress["in_progress_files"].remove(pdf_path)
                self.save_processing_state()
                
            return self.paper_details[paper_id]
        
        logger.info(f"[Depth {current_depth}] Processing paper: {paper_id}")
        
        # Check if we've reached max depth
        if self.max_depth >= 0 and current_depth > self.max_depth:
            logger.info(f"Reached maximum depth ({self.max_depth}) for {paper_id}, skipping further processing")
            
            with self.lock:
                if pdf_path in self.current_processing:
                    del self.current_processing[pdf_path]
                
                self.progress["processed_files"].append(pdf_path)
                if pdf_path in self.progress["pending_files"]:
                    self.progress["pending_files"].remove(pdf_path)
                if pdf_path in self.progress["in_progress_files"]:
                    self.progress["in_progress_files"].remove(pdf_path)
                    
            self.save_state()
            return {"depth_limit_reached": True, "path": pdf_path}
        
        try:
            # Update processing status
            with self.lock:
                self.current_processing[pdf_path]["status"] = "extracting_text"
                self.save_processing_state()
            
            # Extract text from PDF
            if self.text_ratio:
                head_ratio, tail_ratio = self.text_ratio
                paper_text = extract_text_by_ratio(pdf_path, head_ratio, tail_ratio)
                logger.info(f"[Depth {current_depth}] Extracted text using ratio {head_ratio}%,{tail_ratio}% from {paper_id}")
            else:
                paper_text = extract_text_with_limit(pdf_path)
                logger.info(f"[Depth {current_depth}] Extracted full text from {paper_id}")
            
            if not paper_text:
                logger.error(f"[Depth {current_depth}] Failed to extract text from {paper_id}")
                
                with self.lock:
                    self.progress["failed_files"].append(pdf_path)
                    if pdf_path in self.progress["pending_files"]:
                        self.progress["pending_files"].remove(pdf_path)
                    if pdf_path in self.progress["in_progress_files"]:
                        self.progress["in_progress_files"].remove(pdf_path)
                    if pdf_path in self.current_processing:
                        del self.current_processing[pdf_path]
                        
                self.save_state()
                return {"error": "Text extraction failed", "path": pdf_path}
            
            # Update processing status
            with self.lock:
                self.current_processing[pdf_path]["status"] = "querying_llm"
                self.current_processing[pdf_path]["text_extraction_time"] = get_timestamp_str()
                self.save_processing_state()
            
            # Check for auto-save
            self.check_autosave()
            
            # Query LLM to extract information
            paper_info = self.llm_service.extract_paper_info(paper_text)
            
            # Check for LLM failure
            if not paper_info or (paper_info.get("llm_usage", {}).get("success") is False and len(paper_info) == 1):
                logger.error(f"[Depth {current_depth}] LLM extraction failed for {paper_id}")
                
                with self.lock:
                    self.progress["failed_files"].append(pdf_path)
                    if pdf_path in self.progress["pending_files"]:
                        self.progress["pending_files"].remove(pdf_path)
                    if pdf_path in self.progress["in_progress_files"]:
                        self.progress["in_progress_files"].remove(pdf_path)
                    if pdf_path in self.current_processing:
                        del self.current_processing[pdf_path]
                        
                self.save_state()
                return {"error": "LLM extraction failed", "path": pdf_path}
            
            # Update processing status
            with self.lock:
                self.current_processing[pdf_path]["status"] = "processing_references"
                self.current_processing[pdf_path]["llm_completion_time"] = get_timestamp_str()
                self.save_processing_state()
            
            # Add metadata
            paper_info["source_pdf"] = pdf_path
            paper_info["processing_timestamp"] = get_timestamp_str()
            paper_info["processing_depth"] = current_depth
            
            # Check for auto-save
            self.check_autosave()
            
            # Process references for download queue
            references = paper_info.get("references", [])
            next_depth = current_depth + 1
            
            # Process each reference
            for ref in references:
                # Add to download queue if title exists 
                if ref.get("ref_title"):
                    should_queue = True
                    
                    # Check if we're at max depth
                    if self.max_depth >= 0 and next_depth > self.max_depth:
                        ref["depth_limit_reached"] = True
                        should_queue = False
                        logger.debug(f"Not queueing reference due to depth limit: {ref.get('ref_title')}")
                    
                    if should_queue:
                        with self.lock:
                            # Lookup DOI for this reference
                            title = ref.get("ref_title")
                            
                            # Check if this reference is already in the queue
                            exists = any(item["title"] == title for item in self.download_queue)
                            
                            if not exists:
                                # Create download queue item
                                queue_item = {
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
                                }
                                
                                self.download_queue.append(queue_item)
            
            # Update statistics
            self.update_stats("processed_count")
            self.update_stats("found_references", len(references))
            
            # Map to correct output directory
            output_dir = self.dir_mapping.get(pdf_path, self.output_dir)
            
            # Add to paper details and update progress
            with self.lock:
                self.paper_details[paper_id] = paper_info
                self.progress["processed_files"].append(pdf_path)
                if pdf_path in self.progress["pending_files"]:
                    self.progress["pending_files"].remove(pdf_path)
                if pdf_path in self.progress["in_progress_files"]:
                    self.progress["in_progress_files"].remove(pdf_path)
                if pdf_path in self.current_processing:
                    del self.current_processing[pdf_path]
            
            # Save state
            self.save_state()
            return paper_info
            
        except Exception as e:
            logger.error(f"[Depth {current_depth}] Error processing {paper_id}: {e}", exc_info=True)
            
            with self.lock:
                self.progress["failed_files"].append(pdf_path)
                if pdf_path in self.progress["pending_files"]:
                    self.progress["pending_files"].remove(pdf_path)
                if pdf_path in self.progress["in_progress_files"]:
                    self.progress["in_progress_files"].remove(pdf_path)
                if pdf_path in self.current_processing:
                    del self.current_processing[pdf_path]
                    
            self.save_state()
            return {"error": str(e), "path": pdf_path}
            
    def lookup_doi_for_queue(self):
        """Look up DOIs for items in the download queue"""
        pending_lookups = [item for item in self.download_queue if item["doi_lookup_status"] == "pending"]
        
        if not pending_lookups:
            logger.info("No pending DOI lookups")
            return
            
        logger.info(f"Looking up DOIs for {len(pending_lookups)} references")
        
        for i, item in enumerate(pending_lookups):
            title = item["title"]
            logger.info(f"[{i+1}/{len(pending_lookups)}] Looking up DOI for: {title}")
            
            # Skip if already has DOI
            if item.get("doi"):
                item["doi_lookup_status"] = "success"
                continue
                
            # Query CrossRef API
            result = get_doi_from_title(title)
            
            # Update item with DOI information
            item["doi_lookup_result"] = result
            item["doi_lookup_time"] = get_timestamp_str()
            
            if result.get("doi"):
                item["doi"] = result["doi"]
                item["doi_lookup_status"] = "success"
                item["doi_match_score"] = result.get("score", 0)
                logger.info(f"✓ Found DOI: {result['doi']} (score: {result.get('score', 0):.2f})")
            else:
                item["doi_lookup_status"] = "failed"
                item["doi_lookup_error"] = result.get("error", "Unknown error")
                logger.warning(f"✗ DOI lookup failed: {result.get('error', 'Unknown error')}")
            
            # Save state periodically
            if (i + 1) % 5 == 0:
                self.save_state()
                
            # Add delay between requests
            if i < len(pending_lookups) - 1:
                time.sleep(2)  # Respect API rate limits
        
        # Update pending downloads after DOI lookup
        self._update_pending_downloads()
        
        # Final save
        self.save_state()
        
        # Count successes
        success_count = sum(1 for item in pending_lookups if item["doi_lookup_status"] == "success")
        logger.info(f"DOI lookup complete: {success_count}/{len(pending_lookups)} successful")
                
    def download_papers_from_queue(self, batch_size=20, batch_delay=300):
        """Download papers that have DOIs from the download queue in batches"""
        # Update pending downloads
        self._update_pending_downloads()
        
        # Filter items with successful DOI lookup that haven't been downloaded yet
        to_download = self.pending_downloads
        
        if not to_download:
            logger.info("No papers to download")
            return
        
        logger.info(f"Downloading {len(to_download)} papers")
        
        # Process in batches to avoid overloading
        for batch_idx in range(0, len(to_download), batch_size):
            batch = to_download[batch_idx:batch_idx+batch_size]
            logger.info(f"Processing batch {batch_idx//batch_size + 1}/{(len(to_download)+batch_size-1)//batch_size} ({len(batch)} papers)")
            
            for i, item in enumerate(batch):
                doi = item["doi"]
                title = item["title"]
                source_pdf = item.get("source_pdf", "")
                
                # Get appropriate download directory based on source paper
                download_dir = self.get_download_dir(source_pdf)
                
                logger.info(f"[{i+1}/{len(batch)}] Downloading: {title} (DOI: {doi})")
                item["download_attempt_time"] = get_timestamp_str()
                
                try:
                    # Download from SciHub with random mirror selection
                    success, local_path, error = download_from_scihub(doi, str(download_dir), delay=self.download_delay)
                    
                    # Update item with download information
                    item["download_status"] = "complete" if success else "failed"
                    item["download_success"] = success
                    item["download_time"] = get_timestamp_str()
                    
                    if success and local_path:
                        item["local_path"] = local_path
                        logger.info(f"✓ Successfully downloaded paper to {local_path}")
                        self.update_stats("downloaded_papers")
                        
                        # Add to successful downloads
                        self.completed_downloads.append(item.copy())
                        
                        # Add to download results
                        self.download_results.append(item.copy())
                        
                        # If we're not at max depth, add to processing queue for next depth
                        if self.max_depth < 0 or item["target_depth"] <= self.max_depth:
                            # Process downloaded paper (may run in parallel during next phase)
                            self.pending_recursive_papers.append({
                                "path": local_path,
                                "depth": item["target_depth"]
                            })
                    else:
                        logger.warning(f"✗ Failed to download paper: {error}")
                        item["error_message"] = error
                        self.update_stats("download_failures")
                        self.failed_downloads.append(item.copy())
                        self.download_results.append(item.copy())
                
                except Exception as e:
                    logger.error(f"Error downloading paper: {e}")
                    item["download_status"] = "error"
                    item["download_error"] = str(e)
                    self.update_stats("download_failures")
                    self.failed_downloads.append(item.copy())
                    self.download_results.append(item.copy())
                
                # Save state periodically
                if (i + 1) % 2 == 0:
                    self.save_state()
                    
                # Add delay between downloads within batch
                if i < len(batch) - 1:
                    time.sleep(self.download_delay)
            
            # Save state after each batch
            self.save_state()
            
            # Add delay between batches if there are more batches to process
            if batch_idx + batch_size < len(to_download):
                logger.info(f"Batch complete, pausing for {batch_delay//60} minutes before next batch...")
                time.sleep(batch_delay)
        
        # Update pending downloads
        self._update_pending_downloads()
        
        # Final save
        self.save_state()
        
        # Count successes from this run
        success_count = sum(1 for item in to_download if item.get("download_success", False))
        logger.info(f"Download complete: {success_count}/{len(to_download)} successful")
    
    def process_references(self, workers=None):
        """Process papers from the download queue up to max depth"""
        if not workers:
            workers = self.max_workers
        
        # Skip if max_depth is 0 (only process initial papers)
        if self.max_depth == 0:
            logger.info("Max depth is 0, skipping references processing")
            return
        
        # Initialize tracker for papers to recursively process
        self.pending_recursive_papers = []
        
        # Lookup DOIs for references in queue
        self.lookup_doi_for_queue()
        
        # Download papers for references with DOIs
        self.download_papers_from_queue()
        
        # Retry failed downloads if any
        if self.failed_downloads:
            logger.info(f"Retrying {len(self.failed_downloads)} failed downloads...")
            # Get download directory
            download_dir = self.get_download_dir()
            still_failed, newly_successful = retry_failed_downloads(
                self.failed_downloads, 
                str(download_dir), 
                delay=self.download_delay
            )
            
            # Update stats and lists
            self.failed_downloads = still_failed
            
            # Update download results with newly successful items
            for item in newly_successful:
                # Add to successful downloads
                self.completed_downloads.append(item)
                # Add to pending recursive processing
                if self.max_depth < 0 or item["target_depth"] <= self.max_depth:
                    self.pending_recursive_papers.append({
                        "path": item["local_path"],
                        "depth": item["target_depth"]
                    })
                    
            # Update stats
            self.update_stats("downloaded_papers", len(newly_successful))
            self.update_stats("download_failures", -len(newly_successful))
            
            self.save_state()
            
        # Process downloaded papers recursively
        if self.pending_recursive_papers:
            logger.info(f"Processing {len(self.pending_recursive_papers)} downloaded reference papers")
            
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(
                        self.process_paper, 
                        item["path"], 
                        item["depth"]
                    ) for item in self.pending_recursive_papers
                ]
                
                # Wait for all to complete
                for i, future in enumerate(futures):
                    try:
                        result = future.result()
                        logger.info(f"Completed recursive paper [{i+1}/{len(futures)}]: {result.get('title', 'Unknown')}")
                    except Exception as e:
                        logger.error(f"Error in recursive paper processing: {e}")
            
            # Now recursively process newly found references
            logger.info("Starting recursive reference processing")
            self.process_references(workers)
    
    def process_all_papers(self, subdir=None):
        """Process all pending papers in parallel."""
        pending_files = self.scan_input_directory(subdir)
        
        if not pending_files:
            # Check if we have pending downloads that need to be processed
            if self.pending_downloads:
                logger.info(f"No new papers to process, but found {len(self.pending_downloads)} pending downloads")
                self.process_references()
                return
            
            # Check for failed downloads that could be retried
            if self.failed_downloads:
                logger.info(f"No new papers to process, but found {len(self.failed_downloads)} failed downloads to retry")
                self.retry_failed_downloads_only()
                return
                
            logger.info("No new papers to process and no pending downloads")
            return
        
        logger.info(f"Starting processing of {len(pending_files)} papers with {self.max_workers} workers")
        self.stats["status"] = "processing"
        self.stats["max_depth"] = self.max_depth
        self.save_state()
        
        start_time = time.time()
        
        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(self.process_paper, pdf_path, 0) for pdf_path in pending_files]
                
                # Process as they complete
                for i, future in enumerate(futures):
                    try:
                        result = future.result()
                        logger.info(f"Completed {i+1}/{len(futures)}: {result.get('title', 'Unknown title')}")
                    except Exception as e:
                        logger.error(f"Worker failed: {e}")
                
            # Process references if max_depth allows
            if self.max_depth != 0:
                self.process_references()
                        
            # Update final stats
            end_time = time.time()
            runtime = end_time - start_time
            hours, remainder = divmod(runtime, 3600)
            minutes, seconds = divmod(remainder, 60)
            runtime_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
            
            self.stats["status"] = "completed"
            self.stats["end_time"] = get_timestamp_str()
            self.stats["runtime"] = runtime_str
            
        except KeyboardInterrupt:
            logger.info("Process interrupted by user. Saving state.")
            self.stats["status"] = "interrupted"
            self.stats["end_time"] = get_timestamp_str()
        finally:
            # Save final state
            self.save_state()
            
            # Print summary
            logger.info("=" * 50)
            logger.info(f"Processing complete. Processed {self.stats.get('processed_count', 0)} papers.")
            logger.info(f"Found {self.stats.get('found_references', 0)} references.")
            logger.info(f"Downloaded {self.stats.get('downloaded_papers', 0)} reference papers.")
            logger.info(f"Download failures: {self.stats.get('download_failures', 0)}")
            logger.info("=" * 50)
            
    def retry_failed_downloads_only(self):
        """Standalone function to retry just the failed downloads"""
        if not self.failed_downloads:
            logger.info("No failed downloads to retry")
            return
            
        logger.info(f"Retrying {len(self.failed_downloads)} failed downloads...")
        download_dir = self.get_download_dir()
        still_failed, newly_successful = retry_failed_downloads(
            self.failed_downloads, 
            str(download_dir), 
            delay=self.download_delay, 
            max_retries=5
        )
                                             
        # Update stats
        newly_successful_count = len(newly_successful)
        if newly_successful_count > 0:
            self.update_stats("downloaded_papers", newly_successful_count)
            self.update_stats("download_failures", -newly_successful_count)
            
            # Add newly successful downloads to pending recursive papers
            pending_recursive = []
            for item in newly_successful:
                if self.max_depth < 0 or item["target_depth"] <= self.max_depth:
                    pending_recursive.append({
                        "path": item["local_path"],
                        "depth": item["target_depth"]
                    })
            
            # Update download results to reflect successful retries
            for item in self.download_results:
                for success in newly_successful:
                    if item.get("doi") == success.get("doi"):
                        item.update({
                            "download_success": True,
                            "download_status": "complete",
                            "local_path": success.get("local_path"),
                            "retry_success": True,
                            "retry_time": success.get("retry_time")
                        })
            
            # Process newly downloaded papers if any
            if pending_recursive:
                logger.info(f"Processing {len(pending_recursive)} newly downloaded papers")
                for item in pending_recursive:
                    self.process_paper(item["path"], item["depth"])
        
        # Update failed downloads list
        self.failed_downloads = still_failed
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
        if head < 0 or tail < 0 or head + tail > 100:
            return None
        return (head, tail)
    except ValueError:
        return None

def main():
    # Define command line arguments
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
    parser.add_argument('--retry-downloads', action='store_true', help='Only retry failed downloads')
    parser.add_argument('--batch-size', type=int, default=20, help='Number of papers to download in each batch')
    parser.add_argument('--batch-delay', type=int, default=300, help='Delay between download batches in seconds')
    
    args = parser.parse_args()
    
    # Most important: configure paths before any other operations
    if args.subdir:
        # When using subdir, we need to configure output with correct subdirectory
        config.configure_paths(args.input_dir, args.output_dir, args.subdir)
    else:
        config.configure_paths(args.input_dir, args.output_dir)
        
    paths = config.get_configured_paths()
    logger.info(f"Configured paths: Input={paths['input_dir']}, Output={paths['output_dir']}")
    
    # Fix API URL if needed
    api_base = validate_api_url(args.api_base)
    
    # Parse text ratio parameter
    text_ratio = parse_text_ratio(args.text_ratio)
    if args.text_ratio and text_ratio is None:
        logger.warning(f"Invalid text ratio '{args.text_ratio}'. Format should be 'head,tail' (e.g., '10,30'). Using full text instead.")
    
    # Collect LLM parameters
    llm_params = {
        'temperature': args.temperature,
        'top_p': args.top_p,
        'top_k': args.top_k,
        'presence_penalty': args.presence_penalty,
        'max_tokens': args.max_tokens
    }
    # Remove None values
    llm_params = {k: v for k, v in llm_params.items() if v is not None}
    
    # Initialize and run parser
    try:
        paper_parser = PaperParser(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            api_base=api_base,
            api_key=args.api_key,
            model=args.model,
            max_workers=args.max_workers,
            max_depth=args.max_depth,
            text_ratio=text_ratio,
            save_interval=args.save_interval,
            download_delay=args.download_delay,
            **llm_params
        )
        
        if args.retry_downloads:
            # Just retry failed downloads
            logger.info("Running in retry-downloads mode")
            paper_parser.retry_failed_downloads_only()
        else:
            # Normal operation - process papers and references
            logger.info(f"Starting paper parsing with max depth: {args.max_depth}...")
            paper_parser.process_all_papers(args.subdir)
        
    except KeyboardInterrupt:
        logger.info("Parser stopped by user")
    except Exception as e:
        logger.error(f"Parser stopped due to error: {e}", exc_info=True)

if __name__ == "__main__":
    main()
