# src/config.py - Dynamic path configuration

import os
from pathlib import Path

# Variables to store actual paths - will be set from command line args
INPUT_DIR = None
OUTPUT_DIR = None
DOWNLOAD_DIR = None
SUBDIR = None

# State file paths - will be updated when output directory is set
PAPER_DETAILS_FILE = None
STATS_FILE = None
DOWNLOAD_QUEUE_FILE = None
DOWNLOAD_RESULTS_FILE = None
PROGRESS_FILE = None
PROCESSING_STATE_FILE = None

# Initialization status - not yet configured
PATHS_CONFIGURED = False

def configure_paths(input_dir, output_dir, subdir=None):
    """Configure all paths - must be called before using any paths"""
    global INPUT_DIR, OUTPUT_DIR, DOWNLOAD_DIR, SUBDIR
    global PAPER_DETAILS_FILE, STATS_FILE, DOWNLOAD_QUEUE_FILE, DOWNLOAD_RESULTS_FILE
    global PROGRESS_FILE, PROCESSING_STATE_FILE, PATHS_CONFIGURED
    
    # Set main directories
    INPUT_DIR = Path(input_dir)
    OUTPUT_DIR = Path(output_dir)
    SUBDIR = subdir
    
    # If subdir is specified, adjust input and output paths
    if subdir:
        input_subdir = INPUT_DIR / subdir
        output_subdir = OUTPUT_DIR / subdir
        
        # Use subdir for actual processing
        if input_subdir.exists():
            # For output, we create the same subdirectory structure
            output_subdir.mkdir(parents=True, exist_ok=True)
            
            # Use the subdirectory as our actual output location
            actual_output_dir = output_subdir
            DOWNLOAD_DIR = str(output_subdir / "downloads")
        else:
            raise ValueError(f"Specified subdirectory {subdir} not found in {INPUT_DIR}")
    else:
        actual_output_dir = OUTPUT_DIR
        DOWNLOAD_DIR = str(OUTPUT_DIR / "downloads")
    
    # Ensure directories exist
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(actual_output_dir, exist_ok=True)
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    
    # Set file paths using the actual output directory
    PAPER_DETAILS_FILE = str(Path(actual_output_dir) / "paper_details.json")
    STATS_FILE = str(Path(actual_output_dir) / "crawler_stats.json")
    DOWNLOAD_QUEUE_FILE = str(Path(actual_output_dir) / "download_queue.json")
    DOWNLOAD_RESULTS_FILE = str(Path(actual_output_dir) / "download_results.json")
    PROGRESS_FILE = str(Path(actual_output_dir) / "processing_progress.json")
    PROCESSING_STATE_FILE = str(Path(actual_output_dir) / "processing_state.json")
    
    # Mark as configured
    PATHS_CONFIGURED = True
    
    return {
        "input_dir": INPUT_DIR,
        "output_dir": actual_output_dir,
        "download_dir": DOWNLOAD_DIR,
        "paper_details_file": PAPER_DETAILS_FILE,
        "stats_file": STATS_FILE,
        "download_queue_file": DOWNLOAD_QUEUE_FILE,
        "download_results_file": DOWNLOAD_RESULTS_FILE,
        "progress_file": PROGRESS_FILE,
        "processing_state_file": PROCESSING_STATE_FILE
    }

# Decorator to ensure paths are configured
def requires_configured_paths(func):
    """Decorator to ensure paths are configured before access"""
    def wrapper(*args, **kwargs):
        if not PATHS_CONFIGURED:
            raise RuntimeError("Must call configure_paths() to set paths before access")
        return func(*args, **kwargs)
    return wrapper

# Check path configuration status
@requires_configured_paths
def get_configured_paths():
    """Get all currently configured paths"""
    return {
        "input_dir": INPUT_DIR,
        "output_dir": OUTPUT_DIR,
        "download_dir": DOWNLOAD_DIR,
        "paper_details_file": PAPER_DETAILS_FILE,
        "stats_file": STATS_FILE,
        "download_queue_file": DOWNLOAD_QUEUE_FILE,
        "download_results_file": DOWNLOAD_RESULTS_FILE,
        "progress_file": PROGRESS_FILE,
        "processing_state_file": PROCESSING_STATE_FILE
    }

# Default LLM prompt remains unchanged
DEFAULT_PROMPT = """
Extract the following information from this academic paper as JSON:

- title: The paper's full title
- authors: List of authors' names
- affiliations: List of author affiliations
- abstract: The paper's abstract
- references: List of references with the following details for each reference:
  - ref_title: Full title of the referenced paper (IMPORTANT: provide complete titles)
  - ref_authors: List of authors of the referenced paper
  - ref_year: Publication year
  - ref_venue: Publication venue (journal/conference)

IMPORTANT INSTRUCTIONS:
1. Pay special attention to the References or Bibliography section at the end of the paper
2. Extract ALL references you can find, even if the paper text is truncated
3. For each reference, ensure you capture the complete title - this is crucial
4. Return ONLY valid JSON with no additional text or formatting
5. If you cannot find certain information, use empty strings or arrays rather than omitting fields

PAPER TEXT:
{paper_text}
"""
