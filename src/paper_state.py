# paper_state.py - State management for paper crawler

import json
import time
import logging
import os
from pathlib import Path
import threading
import datetime

logger = logging.getLogger(__name__)

class PaperState:
    def __init__(self, output_dir):
        """Initialize state manager with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # State files
        self.details_file = self.output_dir / "paper_details.json"
        self.stats_file = self.output_dir / "crawler_stats.json"
        
        # Initialize state
        self.details = self._load_json(self.details_file, {})
        self.stats = self._load_json(self.stats_file, {
            "processed_count": 0,
            "download_success_count": 0,
            "download_fail_count": 0,
            "reference_count": 0,
            "start_time": self._get_timestamp_str(),
            "last_update": self._get_timestamp_str()
        })
        
        self.lock = threading.Lock()
    
    def _get_timestamp_str(self):
        """Get current time as string in YYYY-MM-DD HH:MM:SS format."""
        return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    def _load_json(self, file_path, default=None):
        """Load JSON file or return default if not exists/invalid."""
        if file_path.exists():
            try:
                with open(file_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Corrupted JSON file {file_path}. Using default.")
        return default if default is not None else {}
    
    def _save_json(self, data, file_path):
        """Save data to JSON file with pretty printing."""
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def update_paper_details(self, paper_id, details):
        """Update details for a specific paper."""
        with self.lock:
            self.details[paper_id] = details
            self._save_json(self.details, self.details_file)
    
    def get_paper_details(self, paper_id):
        """Get details for a specific paper."""
        with self.lock:
            return self.details.get(paper_id, {})
    
    def update_stats(self, stats_update):
        """Update crawler statistics."""
        with self.lock:
            self.stats.update(stats_update)
            self.stats["last_update"] = self._get_timestamp_str()
            self._save_json(self.stats, self.stats_file)
    
    def increment_stats(self, key, amount=1):
        """Increment a specific statistic."""
        with self.lock:
            self.stats[key] = self.stats.get(key, 0) + amount
            self.stats["last_update"] = self._get_timestamp_str()
            self._save_json(self.stats, self.stats_file)
    
    def paper_exists(self, paper_id):
        """Check if paper has been processed."""
        with self.lock:
            return paper_id in self.details
    
    def get_all_papers(self):
        """Get all paper details."""
        with self.lock:
            return dict(self.details)
    
    def get_stats(self):
        """Get current statistics."""
        with self.lock:
            return dict(self.stats)
