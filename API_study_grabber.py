"""
ClinicalTrials.gov API Wrapper (Cleaned)
Minimal, essential functions only.
"""

import requests
from typing import Dict, Any, Optional


class ClinicalTrialsAPI:
    """Simple wrapper for ClinicalTrials.gov API v2."""
    
    url = "https://clinicaltrials.gov/api/v2"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'ClinicalTrials-Python-Client/1.0'
        })
    
    def get_study(self, nct_id: str) -> Dict[str, Any]:
        """
        Fetch a single study by NCT ID.
        
        Args:
            nct_id: NCT ID (e.g., 'NCT04280705')
        
        Returns:
            Study data dictionary
        """
        url = f"{self.url}/studies/{nct_id}"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error fetching {nct_id}: {e}")
    
    def search_studies(self, query: str, page_size: int = 1000) -> Dict[str, Any]:
        """
        Search for studies.
        
        Args:
            query: Search term
            page_size: Results per page (max 1000)
        
        Returns:
            Search results
        """
        url = f"{self.url}/studies"
        params = {
            'query.term': query,
            'pageSize': min(page_size, 1000)
        }
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Search error: {e}")