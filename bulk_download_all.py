"""
Downloads ALL Clinical Trials from ClinicalTrials.gov, uses downloads in batches of 1000 pages since the API limit is
1000 per request
"""

import requests
import zipfile
import os
import json
import time
from datetime import datetime
from typing import List, Dict, Any
import xml.etree.ElementTree as ET
from pathlib import Path

from similarity import TrialSimilarityEngine
from API_study_grabber import ClinicalTrialsAPI


class BulkTrialDownloader:
    """
    Download all trials using the bulk download feature.
    This is MUCH faster than API pagination.
    """
    
    # clinicalTrials.gov api link
    ct_url = "https://clinicaltrials.gov/api/v2/studies"
    
    def __init__(self, output_dir='all_trials_data'):
        self.output_dir = output_dir
        ## put the output into a folder
        os.makedirs(output_dir, exist_ok=True)
    
    def download_all_trials_bulk(self, format='json'):
        """
        Download all trials in bulk using the API's pagination.
        
        The API v2 allows downloading in chunks of 1000 trials.
        We'll paginate through all of them.
        
        Args:
            format: 'json' or 'csv'
        
        Returns:
            List of all studies
        """        
        all_studies = []
        page_size = 1000  # api cap DO NOT CHANGE
        page_token = None
        page_num = 0
        
        start_time = time.time()
        
        while True:
            page_num += 1            
            try:
                params = {
                    'pageSize': page_size,
                    'format': format
                }
                
                if page_token:
                    params['pageToken'] = page_token
                
                response = requests.get(self.ct_url, params=params)
                response.raise_for_status()
                
                data = response.json()
                
                # extract studies
                studies = data.get('studies', [])
                all_studies.extend(studies)
                
                print(f"Downloaded {len(studies)} trials")
                print(f"Total so far: {len(all_studies):,}")
                
                # check if there are more pages
                next_page_token = data.get('nextPageToken')
                
                if not next_page_token or len(studies) == 0:
                    print("\n✓ end")
                    break
                
                page_token = next_page_token
                
                # save checkpoint every 10 pages/10,000 trials incase fo error
                if page_num % 10 == 0:
                    self.save_checkpoint(all_studies, page_num)
                
                # rate limiting
                time.sleep(2)
            
            ## create exception that retrys if fails
            except Exception as e:
                print(f"\n✗ Error on page {page_num}: {e}")
                self.save_checkpoint(all_studies, page_num)
                
                # retry after giving api a break
                time.sleep(30)
                continue
        
        elapsed = time.time() - start_time
        print(f"Total trials downloaded: {len(all_studies):,}")
        print(f"Time elapsed: {elapsed/3600:.1f} hours")
        
        # save final data
        self.save_all_trials(all_studies)
        
        return all_studies
    
    def save_checkpoint(self, studies, page_num):
        """Save intermediate checkpoint."""
        filename = os.path.join(self.output_dir, f'checkpoint_page_{page_num}.json')
        with open(filename, 'w') as f:
            json.dump(studies, f)
        print(f"Checkpoint saved: {filename}")
    
    def save_all_trials(self, studies):
        """Save all trials to final file."""
        filename = os.path.join(self.output_dir, 'all_trials_complete.json')  
        with open(filename, 'w') as f:
            json.dump(studies, f)
        
        return filename
    
    def load_from_checkpoint(self, checkpoint_file):
        """Load trials from a checkpoint file."""
        with open(checkpoint_file, 'r') as f:
            studies = json.load(f)
        return studies


class TrialIndexBuilder:
    """
    Build similarity index from downloaded trials.
    
    Since we have 500k trials, we'll build the index in batches
    to avoid memory issues.
    """
    
    def __init__(self):
        self.engine = TrialSimilarityEngine()
    
    def build_index_from_file(self, trials_file):
        """
        Build index from downloaded trials file.
        
        Args:
            trials_file: Path to JSON file with all trials
            sample_size: If provided, only use this many trials (for testing)
        """
        
        # load trials
        with open(trials_file, 'r') as f:
            all_studies = json.load(f)
        
        print(f"loaded {len(all_studies):,} trials")
        
        # build index       
        start_time = time.time()
        self.engine.build_index(all_studies)
        elapsed = time.time() - start_time
        print(f"Time elapsed: {elapsed/60:.1f} minutes")
        
        return self.engine
    
    def save_index(self, filename='trial_similarity_index_complete.pkl'):
        """Save the complete index."""
        self.engine.save_index(filename)
        

def resume_download_from_checkpoint(checkpoint_dir='all_trials_data'):
    """
    kept erroring, need to create checkpoints and loading checkpoints to resume from
    Resume download from the latest checkpoint.
    """
    checkpoint_files = list(Path(checkpoint_dir).glob('checkpoint_page_*.json'))
    
    if not checkpoint_files:
        print("No checkpoints found. Starting fresh download.")
        return None
    
    # find latest checkpoint
    latest = max(checkpoint_files, key=lambda p: int(p.stem.split('_')[-1]))   
    downloader = BulkTrialDownloader(checkpoint_dir)
    studies = downloader.load_from_checkpoint(latest)
    
    print(f"\nResuming from {len(studies):,} trials already downloaded")
    return studies


# exec

if __name__ == "__main__":
       
    print("\nOptions:")
    print("  1. Download and build index (2-5 hours total)")
    print("  2. Build index from existing download")
    print("  3. Resume from checkpoint")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == '1':
        # Download and build index
        downloader = BulkTrialDownloader()
        trials = downloader.download_all_trials_bulk()
        
        # Build index        
        builder = TrialIndexBuilder()
        builder.build_index_from_file('all_trials_data/all_trials_complete.json')
        builder.save_index('trial_similarity_index_complete.pkl')
        
    
    elif choice == '2':
        # build index from existing file
        trials_file = input("path to trials JSON file: ").strip()
        
        if not os.path.exists(trials_file):
            print(f"Error: {trials_file} not found!")
        else:
            builder = TrialIndexBuilder()
            builder.build_index_from_file(trials_file)
            builder.save_index('trial_similarity_index_complete.pkl')
    
    elif choice == '3':
        # Resume from checkpoint
        studies = resume_download_from_checkpoint()
    
    else:
        print("Invalid choice!")
    
    print("finish")