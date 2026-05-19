"""
PubMed API Connector for Clinical Trial Matching

This module connects clinical trials (via NCT ID) to their published
research papers on PubMed. It uses multiple matching strategies:
1. Direct NCT ID search in PubMed
2. Fuzzy title matching as fallback
3. Author + condition matching for additional validation

Also provides outcome extraction and summarization capabilities.
"""

import requests
import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import re
import time
import json
from datetime import datetime


@dataclass
class PubMedArticle:
    """Represents a PubMed article with relevant fields."""
    pmid: str
    title: str
    abstract: str
    authors: List[str]
    journal: str
    pub_date: str
    doi: Optional[str]
    pmc_id: Optional[str]
    keywords: List[str]
    mesh_terms: List[str]
    publication_types: List[str]
    nct_ids_mentioned: List[str]  # NCT IDs found in the article
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'pmid': self.pmid,
            'title': self.title,
            'abstract': self.abstract,
            'authors': self.authors,
            'journal': self.journal,
            'pub_date': self.pub_date,
            'doi': self.doi,
            'pmc_id': self.pmc_id,
            'keywords': self.keywords,
            'mesh_terms': self.mesh_terms,
            'publication_types': self.publication_types,
            'nct_ids_mentioned': self.nct_ids_mentioned,
            'pubmed_url': f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/",
            'full_text_url': f"https://www.ncbi.nlm.nih.gov/pmc/articles/{self.pmc_id}/" if self.pmc_id else None
        }


@dataclass
class MatchResult:
    """Represents a trial-to-publication match with confidence scoring."""
    nct_id: str
    article: PubMedArticle
    match_method: str  # 'direct_nctid', 'fuzzy_title', 'author_condition'
    confidence_score: float  # 0.0 to 1.0
    match_details: Dict[str, Any]  # Details about why this match was made
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'nct_id': self.nct_id,
            'article': self.article.to_dict(),
            'match_method': self.match_method,
            'confidence_score': self.confidence_score,
            'match_details': self.match_details
        }


class PubMedConnector:
    """
    Connects clinical trials to PubMed publications.
    
    Uses NCBI E-utilities API:
    - ESearch: Search PubMed database
    - EFetch: Retrieve article details
    - ELink: Find related articles
    """
    
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    
    def __init__(self, email: str = "your_email@example.com", api_key: Optional[str] = None):
        """
        Initialize PubMed connector.
        
        Args:
            email: Required by NCBI for API usage tracking
            api_key: Optional NCBI API key (increases rate limit from 3 to 10 req/sec)
        """
        self.email = email
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'ClinicalTrialMatcher/1.0'
        })
        self._last_request_time = 0
        self._min_request_interval = 0.34 if api_key else 1.0  # Rate limiting
    
    def _rate_limit(self):
        """Ensure we don't exceed NCBI rate limits."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()
    
    def _build_params(self, **kwargs) -> Dict[str, str]:
        """Build request parameters with standard fields."""
        params = {
            'email': self.email,
            'tool': 'ClinicalTrialMatcher',
            **kwargs
        }
        if self.api_key:
            params['api_key'] = self.api_key
        return params
    
    # =========================================================================
    # CORE SEARCH METHODS
    # =========================================================================
    
    def search_by_nct_id(self, nct_id: str) -> List[str]:
        """
        Search PubMed for articles mentioning a specific NCT ID.
        
        Args:
            nct_id: Clinical trial NCT ID (e.g., 'NCT04280705')
        
        Returns:
            List of PMIDs
        """
        self._rate_limit()
        
        # Search in all fields for the NCT ID
        # Also search secondary source ID field where NCT IDs are often registered
        query = f'"{nct_id}"[All Fields] OR "{nct_id}"[Secondary Source ID]'
        
        url = f"{self.BASE_URL}/esearch.fcgi"
        params = self._build_params(
            db='pubmed',
            term=query,
            retmax=100,
            retmode='json'
        )
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            pmids = data.get('esearchresult', {}).get('idlist', [])
            return pmids
            
        except Exception as e:
            print(f"Error searching PubMed for {nct_id}: {e}")
            return []
    
    def search_by_title(self, title: str, max_results: int = 20) -> List[str]:
        """
        Search PubMed by article title.
        
        Args:
            title: Search title (will be cleaned and processed)
            max_results: Maximum number of results
        
        Returns:
            List of PMIDs
        """
        self._rate_limit()
        
        # Clean title for better matching
        clean_title = self._clean_title_for_search(title)
        
        url = f"{self.BASE_URL}/esearch.fcgi"
        params = self._build_params(
            db='pubmed',
            term=f'{clean_title}[Title]',
            retmax=max_results,
            retmode='json'
        )
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            return data.get('esearchresult', {}).get('idlist', [])
            
        except Exception as e:
            print(f"Error searching PubMed by title: {e}")
            return []
    
    def search_by_trial_details(self, 
                                 conditions: List[str],
                                 interventions: List[str],
                                 phase: str = None,
                                 max_results: int = 50) -> List[str]:
        """
        Search PubMed using trial conditions and interventions.
        
        Args:
            conditions: List of medical conditions
            interventions: List of interventions/drugs
            phase: Trial phase (optional)
            max_results: Maximum results
        
        Returns:
            List of PMIDs
        """
        self._rate_limit()
        
        # Build a combined query
        query_parts = []
        
        if conditions:
            cond_query = ' OR '.join([f'"{c}"[MeSH Terms]' for c in conditions[:3]])
            query_parts.append(f'({cond_query})')
        
        if interventions:
            int_query = ' OR '.join([f'"{i}"[Title/Abstract]' for i in interventions[:3]])
            query_parts.append(f'({int_query})')
        
        # Limit to clinical trial publications
        query_parts.append('("Clinical Trial"[Publication Type] OR "Randomized Controlled Trial"[Publication Type])')
        
        query = ' AND '.join(query_parts)
        
        url = f"{self.BASE_URL}/esearch.fcgi"
        params = self._build_params(
            db='pubmed',
            term=query,
            retmax=max_results,
            retmode='json',
            sort='relevance'
        )
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            return data.get('esearchresult', {}).get('idlist', [])
            
        except Exception as e:
            print(f"Error searching PubMed by trial details: {e}")
            return []
    
    # =========================================================================
    # ARTICLE FETCHING
    # =========================================================================
    
    def fetch_articles(self, pmids: List[str]) -> List[PubMedArticle]:
        """
        Fetch full article details for a list of PMIDs.
        
        Args:
            pmids: List of PubMed IDs
        
        Returns:
            List of PubMedArticle objects
        """
        if not pmids:
            return []
        
        self._rate_limit()
        
        url = f"{self.BASE_URL}/efetch.fcgi"
        params = self._build_params(
            db='pubmed',
            id=','.join(pmids),
            rettype='xml',
            retmode='xml'
        )
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            return self._parse_pubmed_xml(response.text)
            
        except Exception as e:
            print(f"Error fetching articles: {e}")
            return []
    
    def _parse_pubmed_xml(self, xml_text: str) -> List[PubMedArticle]:
        """Parse PubMed XML response into PubMedArticle objects."""
        articles = []
        
        try:
            root = ET.fromstring(xml_text)
            
            for article_elem in root.findall('.//PubmedArticle'):
                try:
                    article = self._parse_single_article(article_elem)
                    if article:
                        articles.append(article)
                except Exception as e:
                    print(f"Error parsing article: {e}")
                    continue
            
        except ET.ParseError as e:
            print(f"XML parsing error: {e}")
        
        return articles
    
    def _parse_single_article(self, article_elem: ET.Element) -> Optional[PubMedArticle]:
        """Parse a single PubmedArticle XML element."""
        medline = article_elem.find('.//MedlineCitation')
        if medline is None:
            return None
        
        # PMID
        pmid_elem = medline.find('.//PMID')
        pmid = pmid_elem.text if pmid_elem is not None else ''
        
        # Article details
        article = medline.find('.//Article')
        if article is None:
            return None
        
        # Title
        title_elem = article.find('.//ArticleTitle')
        title = self._get_text_content(title_elem) if title_elem is not None else ''
        
        # Abstract
        abstract_parts = []
        abstract_elem = article.find('.//Abstract')
        if abstract_elem is not None:
            for abstract_text in abstract_elem.findall('.//AbstractText'):
                label = abstract_text.get('Label', '')
                text = self._get_text_content(abstract_text)
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)
        abstract = ' '.join(abstract_parts)
        
        # Authors
        authors = []
        for author in article.findall('.//Author'):
            last_name = author.find('LastName')
            fore_name = author.find('ForeName')
            if last_name is not None:
                name = last_name.text or ''
                if fore_name is not None and fore_name.text:
                    name = f"{fore_name.text} {name}"
                authors.append(name)
        
        # Journal
        journal_elem = article.find('.//Journal/Title')
        journal = journal_elem.text if journal_elem is not None else ''
        
        # Publication date
        pub_date = self._extract_pub_date(article)
        
        # DOI
        doi = None
        for article_id in article_elem.findall('.//ArticleId'):
            if article_id.get('IdType') == 'doi':
                doi = article_id.text
                break
        
        # PMC ID
        pmc_id = None
        for article_id in article_elem.findall('.//ArticleId'):
            if article_id.get('IdType') == 'pmc':
                pmc_id = article_id.text
                break
        
        # Keywords
        keywords = []
        for keyword in medline.findall('.//KeywordList/Keyword'):
            if keyword.text:
                keywords.append(keyword.text)
        
        # MeSH terms
        mesh_terms = []
        for mesh in medline.findall('.//MeshHeadingList/MeshHeading/DescriptorName'):
            if mesh.text:
                mesh_terms.append(mesh.text)
        
        # Publication types
        pub_types = []
        for pub_type in article.findall('.//PublicationTypeList/PublicationType'):
            if pub_type.text:
                pub_types.append(pub_type.text)
        
        # Find NCT IDs mentioned in the article
        nct_ids = self._extract_nct_ids(title + ' ' + abstract)
        
        # Also check DataBankList for registered trial IDs
        for databank in article.findall('.//DataBankList/DataBank'):
            for accession in databank.findall('.//AccessionNumber'):
                if accession.text and accession.text.startswith('NCT'):
                    nct_ids.append(accession.text)
        
        # Deduplicate
        nct_ids = list(set(nct_ids))
        
        return PubMedArticle(
            pmid=pmid,
            title=title,
            abstract=abstract,
            authors=authors,
            journal=journal,
            pub_date=pub_date,
            doi=doi,
            pmc_id=pmc_id,
            keywords=keywords,
            mesh_terms=mesh_terms,
            publication_types=pub_types,
            nct_ids_mentioned=nct_ids
        )
    
    def _get_text_content(self, elem: ET.Element) -> str:
        """Extract all text content from an element, including nested elements."""
        return ''.join(elem.itertext()).strip()
    
    def _extract_pub_date(self, article: ET.Element) -> str:
        """Extract publication date from article."""
        # Try ArticleDate first (electronic publication)
        article_date = article.find('.//ArticleDate')
        if article_date is not None:
            year = article_date.find('Year')
            month = article_date.find('Month')
            day = article_date.find('Day')
            if year is not None:
                date_str = year.text
                if month is not None:
                    date_str += f"-{month.text.zfill(2)}"
                    if day is not None:
                        date_str += f"-{day.text.zfill(2)}"
                return date_str
        
        # Fall back to Journal PubDate
        pub_date = article.find('.//Journal/JournalIssue/PubDate')
        if pub_date is not None:
            year = pub_date.find('Year')
            month = pub_date.find('Month')
            if year is not None:
                date_str = year.text
                if month is not None:
                    # Month might be text (e.g., "Jan") or number
                    date_str += f" {month.text}"
                return date_str
        
        return ''
    
    def _extract_nct_ids(self, text: str) -> List[str]:
        """Extract NCT IDs from text using regex."""
        pattern = r'NCT\d{8}'
        return re.findall(pattern, text, re.IGNORECASE)
    
    def _clean_title_for_search(self, title: str) -> str:
        """Clean a trial title for PubMed search."""
        # Remove common prefixes
        prefixes = [
            r'^A\s+',
            r'^An\s+',
            r'^The\s+',
            r'^Phase\s+[I1-4]+[:/]?\s*',
            r'^Randomized\s+',
            r'^Double-blind\s+',
            r'^Placebo-controlled\s+',
            r'^Multicenter\s+',
            r'^Open-label\s+',
        ]
        
        clean = title
        for prefix in prefixes:
            clean = re.sub(prefix, '', clean, flags=re.IGNORECASE)
        
        # Remove special characters but keep spaces
        clean = re.sub(r'[^\w\s]', ' ', clean)
        
        # Collapse multiple spaces
        clean = re.sub(r'\s+', ' ', clean).strip()
        
        return clean
    
    # =========================================================================
    # MATCHING LOGIC
    # =========================================================================
    
    def find_publications_for_trial(self, 
                                     nct_id: str,
                                     trial_title: str = None,
                                     conditions: List[str] = None,
                                     interventions: List[str] = None,
                                     sponsors: List[str] = None) -> List[MatchResult]:
        """
        Find PubMed publications for a clinical trial using multiple strategies.
        
        Args:
            nct_id: NCT ID of the trial
            trial_title: Trial title for fuzzy matching
            conditions: Trial conditions for broader search
            interventions: Trial interventions/drugs
            sponsors: Trial sponsors (for author matching)
        
        Returns:
            List of MatchResult objects, sorted by confidence
        """
        matches = []
        
        # Strategy 1: Direct NCT ID search (highest confidence)
        pmids = self.search_by_nct_id(nct_id)
        if pmids:
            articles = self.fetch_articles(pmids)
            for article in articles:
                # Verify NCT ID is actually in the article
                if nct_id.upper() in [nid.upper() for nid in article.nct_ids_mentioned]:
                    confidence = 0.95  # Very high confidence for direct match
                else:
                    confidence = 0.80  # Still good if found via search
                
                matches.append(MatchResult(
                    nct_id=nct_id,
                    article=article,
                    match_method='direct_nctid',
                    confidence_score=confidence,
                    match_details={
                        'nct_in_text': nct_id.upper() in [nid.upper() for nid in article.nct_ids_mentioned],
                        'search_query': f'"{nct_id}"'
                    }
                ))
        
        # Strategy 2: Fuzzy title matching (if no direct matches or as supplement)
        if trial_title and (not matches or len(matches) < 3):
            title_matches = self._fuzzy_title_match(
                nct_id, 
                trial_title, 
                conditions or [],
                interventions or []
            )
            
            # Only add if not already matched
            existing_pmids = {m.article.pmid for m in matches}
            for match in title_matches:
                if match.article.pmid not in existing_pmids:
                    matches.append(match)
        
        # Strategy 3: Broader search by conditions + interventions (lowest confidence)
        if (not matches or len(matches) < 2) and (conditions or interventions):
            detail_pmids = self.search_by_trial_details(
                conditions or [],
                interventions or [],
                max_results=30
            )
            
            if detail_pmids:
                articles = self.fetch_articles(detail_pmids[:20])
                existing_pmids = {m.article.pmid for m in matches}
                
                for article in articles:
                    if article.pmid in existing_pmids:
                        continue
                    
                    # Calculate match score based on content overlap
                    score = self._calculate_content_overlap(
                        article, 
                        trial_title or '', 
                        conditions or [],
                        interventions or []
                    )
                    
                    if score > 0.3:  # Minimum threshold
                        matches.append(MatchResult(
                            nct_id=nct_id,
                            article=article,
                            match_method='condition_intervention',
                            confidence_score=min(score, 0.60),  # Cap at 0.60 for this method
                            match_details={
                                'content_overlap_score': score,
                                'conditions_searched': conditions,
                                'interventions_searched': interventions
                            }
                        ))
        
        # Sort by confidence
        matches.sort(key=lambda x: x.confidence_score, reverse=True)
        
        return matches
    
    def _fuzzy_title_match(self,
                           nct_id: str,
                           trial_title: str,
                           conditions: List[str],
                           interventions: List[str]) -> List[MatchResult]:
        """
        Find publications using fuzzy title matching.
        """
        matches = []
        
        # Extract key terms from title
        key_terms = self._extract_key_terms(trial_title)
        
        # Build search query with key terms
        if key_terms:
            query = ' AND '.join([f'"{term}"[Title/Abstract]' for term in key_terms[:4]])
            
            self._rate_limit()
            
            url = f"{self.BASE_URL}/esearch.fcgi"
            params = self._build_params(
                db='pubmed',
                term=query,
                retmax=30,
                retmode='json'
            )
            
            try:
                response = self.session.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                pmids = data.get('esearchresult', {}).get('idlist', [])
                
                if pmids:
                    articles = self.fetch_articles(pmids)
                    
                    for article in articles:
                        # Calculate fuzzy match score
                        title_similarity = fuzz.token_set_ratio(
                            trial_title.lower(),
                            article.title.lower()
                        ) / 100.0
                        
                        # Boost score if conditions/interventions appear
                        content_boost = 0
                        article_text = f"{article.title} {article.abstract}".lower()
                        
                        for condition in conditions:
                            if condition.lower() in article_text:
                                content_boost += 0.05
                        
                        for intervention in interventions:
                            if intervention.lower() in article_text:
                                content_boost += 0.05
                        
                        confidence = min(title_similarity * 0.7 + content_boost, 0.75)
                        
                        if confidence > 0.4:  # Minimum threshold
                            matches.append(MatchResult(
                                nct_id=nct_id,
                                article=article,
                                match_method='fuzzy_title',
                                confidence_score=confidence,
                                match_details={
                                    'title_similarity': title_similarity,
                                    'content_boost': content_boost,
                                    'key_terms_searched': key_terms
                                }
                            ))
                
            except Exception as e:
                print(f"Error in fuzzy title match: {e}")
        
        return matches
    
    def _extract_key_terms(self, title: str) -> List[str]:
        """Extract key terms from a trial title for searching."""
        # Remove common clinical trial terms
        stop_terms = {
            'study', 'trial', 'phase', 'randomized', 'double-blind', 'placebo',
            'controlled', 'multicenter', 'open-label', 'efficacy', 'safety',
            'patients', 'subjects', 'participants', 'treatment', 'therapy',
            'versus', 'compared', 'comparing', 'evaluation', 'assess', 'investigate'
        }
        
        # Simple tokenization
        words = re.findall(r'\b[a-zA-Z]{4,}\b', title.lower())
        
        # Filter and return unique key terms
        key_terms = []
        seen = set()
        for word in words:
            if word not in stop_terms and word not in seen:
                key_terms.append(word)
                seen.add(word)
        
        return key_terms[:6]  # Limit to 6 terms
    
    def _calculate_content_overlap(self,
                                    article: PubMedArticle,
                                    trial_title: str,
                                    conditions: List[str],
                                    interventions: List[str]) -> float:
        """Calculate overlap score between article content and trial details."""
        score = 0.0
        article_text = f"{article.title} {article.abstract} {' '.join(article.mesh_terms)}".lower()
        
        # Title word overlap
        trial_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', trial_title.lower()))
        article_title_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', article.title.lower()))
        
        if trial_words:
            title_overlap = len(trial_words & article_title_words) / len(trial_words)
            score += title_overlap * 0.4
        
        # Condition matching
        for condition in conditions:
            if condition.lower() in article_text:
                score += 0.15
        
        # Intervention matching
        for intervention in interventions:
            if intervention.lower() in article_text:
                score += 0.2
        
        # Bonus for clinical trial publication type
        clinical_types = {'clinical trial', 'randomized controlled trial', 'controlled clinical trial'}
        if any(pt.lower() in clinical_types for pt in article.publication_types):
            score += 0.1
        
        return min(score, 1.0)
    
    # =========================================================================
    # OUTCOME EXTRACTION
    # =========================================================================
    
    def extract_outcomes_from_abstract(self, abstract: str) -> Dict[str, Any]:
        """
        Extract outcome information from an article abstract.
        
        This uses pattern matching to identify results sections
        and key findings. For better results, consider using
        BERT-based extraction (see OutcomeSummarizer class).
        
        Args:
            abstract: Article abstract text
        
        Returns:
            Dictionary with extracted outcome information
        """
        outcomes = {
            'has_results': False,
            'efficacy_statements': [],
            'safety_statements': [],
            'statistical_findings': [],
            'conclusions': []
        }
        
        if not abstract:
            return outcomes
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', abstract)
        
        # Keywords for different outcome types
        efficacy_keywords = [
            'improved', 'reduction', 'reduced', 'decreased', 'increased',
            'significant', 'effective', 'efficacy', 'response rate',
            'overall survival', 'progression-free', 'remission'
        ]
        
        safety_keywords = [
            'adverse', 'side effect', 'toxicity', 'tolerability',
            'safe', 'safety', 'discontinuation', 'serious'
        ]
        
        stat_patterns = [
            r'p\s*[<>=]\s*0\.\d+',  # p-values
            r'HR\s*[=:]\s*\d+\.\d+',  # Hazard ratios
            r'OR\s*[=:]\s*\d+\.\d+',  # Odds ratios
            r'CI\s*[=:,]\s*\d+',  # Confidence intervals
            r'\d+%',  # Percentages
            r'95%\s*CI',  # 95% CI mentions
        ]
        
        conclusion_keywords = [
            'conclude', 'conclusion', 'in summary', 'therefore',
            'these results', 'our findings', 'this study demonstrates'
        ]
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Check for efficacy
            if any(kw in sentence_lower for kw in efficacy_keywords):
                outcomes['efficacy_statements'].append(sentence)
                outcomes['has_results'] = True
            
            # Check for safety
            if any(kw in sentence_lower for kw in safety_keywords):
                outcomes['safety_statements'].append(sentence)
                outcomes['has_results'] = True
            
            # Check for statistical findings
            if any(re.search(pattern, sentence, re.IGNORECASE) for pattern in stat_patterns):
                outcomes['statistical_findings'].append(sentence)
                outcomes['has_results'] = True
            
            # Check for conclusions
            if any(kw in sentence_lower for kw in conclusion_keywords):
                outcomes['conclusions'].append(sentence)
        
        return outcomes


class OutcomeSummarizer:
    """
    Summarizes clinical trial outcomes using various methods.
    
    Supports:
    1. Rule-based extraction (fast, no dependencies, no cost)
    2. Transformer-based summarization (good quality, requires transformers library, runs locally)
    3. Gemini-based summarization (best quality, cheapest API option - $0.10/1M input tokens)
    4. OpenAI-based summarization (high quality, more expensive)
    
    Recommended: Use 'gemini' for best balance of quality and cost.
    """
    
    def __init__(self, method: str = 'rule_based'):
        """
        Initialize summarizer.
        
        Args:
            method: 'rule_based', 'transformer', 'gemini', or 'openai'
        """
        self.method = method
        self._transformer_model = None
        self._gemini_api_key = None
        self._openai_api_key = None
    
    def set_gemini_api_key(self, api_key: str):
        """
        Set API key for Gemini-based summarization.
        
        Get your free API key at: https://aistudio.google.com/app/apikey
        """
        self._gemini_api_key = api_key
    
    def set_openai_api_key(self, api_key: str):
        """Set API key for OpenAI-based summarization."""
        self._openai_api_key = api_key
    
    def set_api_key(self, api_key: str):
        """Legacy method - sets Gemini key for backward compatibility."""
        self._gemini_api_key = api_key
    
    def summarize(self, 
                  abstract: str, 
                  trial_title: str = None,
                  max_length: int = 150) -> Dict[str, Any]:
        """
        Generate a summary of trial outcomes.
        
        Args:
            abstract: Article abstract
            trial_title: Optional trial title for context
            max_length: Maximum summary length in words
        
        Returns:
            Dictionary with summary and metadata
        """
        if self.method == 'rule_based':
            return self._rule_based_summary(abstract, max_length)
        elif self.method == 'transformer':
            return self._transformer_summary(abstract, max_length)
        elif self.method == 'gemini':
            return self._gemini_summary(abstract, trial_title, max_length)
        elif self.method == 'openai':
            return self._openai_summary(abstract, trial_title, max_length)
        elif self.method == 'api':
            # Legacy support - defaults to gemini
            return self._gemini_summary(abstract, trial_title, max_length)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _rule_based_summary(self, abstract: str, max_length: int) -> Dict[str, Any]:
        """Generate summary using rule-based extraction."""
        if not abstract:
            return {
                'summary': 'No abstract available.',
                'method': 'rule_based',
                'confidence': 0.0
            }
        
        sentences = re.split(r'(?<=[.!?])\s+', abstract)
        
        # Score sentences for importance
        scored_sentences = []
        
        importance_keywords = [
            ('conclude', 3),
            ('result', 2),
            ('significant', 2),
            ('effective', 2),
            ('demonstrate', 2),
            ('show', 1.5),
            ('found', 1.5),
            ('improved', 2),
            ('reduced', 2),
            ('primary endpoint', 3),
            ('overall survival', 3),
            ('p <', 2),
            ('p=', 2),
        ]
        
        for sentence in sentences:
            score = 0
            sentence_lower = sentence.lower()
            
            for keyword, weight in importance_keywords:
                if keyword in sentence_lower:
                    score += weight
            
            # Position bonus (conclusions usually at end)
            position = sentences.index(sentence) / len(sentences)
            if position > 0.7:
                score += 1
            
            scored_sentences.append((sentence, score))
        
        # Sort by score and take top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        summary_parts = []
        word_count = 0
        
        for sentence, score in scored_sentences:
            words = len(sentence.split())
            if word_count + words <= max_length:
                summary_parts.append(sentence)
                word_count += words
            if word_count >= max_length:
                break
        
        # Sort by original order
        summary_parts.sort(key=lambda s: sentences.index(s))
        
        return {
            'summary': ' '.join(summary_parts),
            'method': 'rule_based',
            'confidence': min(scored_sentences[0][1] / 5, 1.0) if scored_sentences else 0.0,
            'sentences_analyzed': len(sentences),
            'sentences_selected': len(summary_parts)
        }
    
    def _transformer_summary(self, abstract: str, max_length: int) -> Dict[str, Any]:
        """Generate summary using transformer model."""
        try:
            from transformers import pipeline
            
            if self._transformer_model is None:
                print("Loading summarization model...")
                self._transformer_model = pipeline(
                    "summarization",
                    model="facebook/bart-large-cnn",
                    device=-1  # CPU, use 0 for GPU
                )
            
            # BART has a max input length
            truncated_abstract = abstract[:1024]
            
            result = self._transformer_model(
                truncated_abstract,
                max_length=max_length,
                min_length=30,
                do_sample=False
            )
            
            return {
                'summary': result[0]['summary_text'],
                'method': 'transformer',
                'model': 'facebook/bart-large-cnn',
                'confidence': 0.8
            }
            
        except ImportError:
            print("Transformers library not installed. Falling back to rule-based.")
            return self._rule_based_summary(abstract, max_length)
        except Exception as e:
            print(f"Transformer error: {e}. Falling back to rule-based.")
            return self._rule_based_summary(abstract, max_length)
    
    def _gemini_summary(self, abstract: str, trial_title: str, max_length: int) -> Dict[str, Any]:
        """
        Generate summary using Google Gemini API.
        
        Gemini Flash is extremely cost-effective:
        - Input: $0.10 per 1 million tokens
        - Output: $0.40 per 1 million tokens
        - A typical abstract (~300 tokens) costs about $0.00003 to summarize
        
        This method sends the abstract to Gemini with a carefully crafted prompt
        that instructs the model to extract key clinical outcomes, efficacy data,
        and safety findings in a concise format.
        """
        if not self._gemini_api_key:
            print("No Gemini API key set. Falling back to rule-based.")
            print("Get your free key at: https://aistudio.google.com/app/apikey")
            return self._rule_based_summary(abstract, max_length)
        
        try:
            from google import genai
            from google.genai import types
            
            # Initialize the client with your API key
            client = genai.Client(api_key=self._gemini_api_key)
            
            # Craft a prompt specifically designed for clinical trial outcome extraction
            prompt = f"""You are a clinical research analyst. Summarize the key outcomes from this clinical trial publication in {max_length} words or less.

Focus on:
1. Primary efficacy results (response rates, survival data, primary endpoints)
2. Key statistical findings (p-values, hazard ratios, confidence intervals)
3. Safety profile (adverse events, tolerability)
4. Main conclusions

Be concise and factual. Use specific numbers when available.

Trial: {trial_title or 'Clinical Trial'}

Abstract:
{abstract}

Summary:"""
            
            # Generate the summary using Gemini 2.5 Flash Lite (cost-effective and available)
            response = client.models.generate_content(
                model='gemini-2.5-flash-lite',
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=max_length * 2,
                    temperature=0.2
                )
            )
            
            return {
                'summary': response.text.strip(),
                'method': 'gemini',
                'model': 'gemini-2.5-flash-lite',
                'confidence': 0.90
            }
            
        except ImportError:
            print("Google GenAI library not installed.")
            print("Install with: pip install google-genai")
            return self._rule_based_summary(abstract, max_length)
        except Exception as e:
            print(f"Gemini API error: {e}. Falling back to rule-based.")
            return self._rule_based_summary(abstract, max_length)
    
    def _openai_summary(self, abstract: str, trial_title: str, max_length: int) -> Dict[str, Any]:
        """
        Generate summary using OpenAI API.
        
        Uses GPT-4o-mini which is cost-effective:
        - Input: $0.15 per 1 million tokens  
        - Output: $0.60 per 1 million tokens
        
        Slightly more expensive than Gemini but some users prefer OpenAI.
        """
        if not self._openai_api_key:
            print("No OpenAI API key set. Falling back to rule-based.")
            return self._rule_based_summary(abstract, max_length)
        
        try:
            from openai import OpenAI
            
            # Initialize the OpenAI client
            client = OpenAI(api_key=self._openai_api_key)
            
            prompt = f"""You are a clinical research analyst. Summarize the key outcomes from this clinical trial publication in {max_length} words or less.

Focus on:
1. Primary efficacy results (response rates, survival data, primary endpoints)
2. Key statistical findings (p-values, hazard ratios, confidence intervals)
3. Safety profile (adverse events, tolerability)
4. Main conclusions

Be concise and factual. Use specific numbers when available.

Trial: {trial_title or 'Clinical Trial'}

Abstract:
{abstract}

Summary:"""
            
            # Call the OpenAI API
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Cheapest GPT-4 class model
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_length * 2,
                temperature=0.2
            )
            
            return {
                'summary': response.choices[0].message.content.strip(),
                'method': 'openai',
                'model': 'gpt-4o-mini',
                'confidence': 0.90
            }
            
        except ImportError:
            print("OpenAI library not installed. Install with: pip install openai")
            return self._rule_based_summary(abstract, max_length)
        except Exception as e:
            print(f"OpenAI API error: {e}. Falling back to rule-based.")
            return self._rule_based_summary(abstract, max_length)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def match_trial_to_publications(
    nct_id: str,
    trial_title: str = None,
    conditions: List[str] = None,
    interventions: List[str] = None,
    email: str = "your_email@example.com"
) -> List[Dict[str, Any]]:
    """
    Convenience function to find publications for a trial.
    
    Args:
        nct_id: Trial NCT ID
        trial_title: Trial title
        conditions: Trial conditions
        interventions: Trial interventions
        email: Email for NCBI API
    
    Returns:
        List of match results as dictionaries
    """
    connector = PubMedConnector(email=email)
    matches = connector.find_publications_for_trial(
        nct_id=nct_id,
        trial_title=trial_title,
        conditions=conditions,
        interventions=interventions
    )
    
    return [m.to_dict() for m in matches]


def summarize_trial_outcomes(
    abstract: str,
    method: str = 'rule_based',
    gemini_api_key: str = None,
    openai_api_key: str = None
) -> Dict[str, Any]:
    """
    Convenience function to summarize trial outcomes.
    
    Args:
        abstract: Article abstract
        method: 'rule_based', 'transformer', 'gemini', or 'openai'
        gemini_api_key: API key for Gemini (required if method='gemini')
        openai_api_key: API key for OpenAI (required if method='openai')
    
    Returns:
        Summary dictionary
    
    Example usage with Gemini:
        summary = summarize_trial_outcomes(
            abstract="Background: This study evaluated...",
            method='gemini',
            gemini_api_key='your-api-key-here'
        )
    """
    summarizer = OutcomeSummarizer(method=method)
    
    if gemini_api_key:
        summarizer.set_gemini_api_key(gemini_api_key)
    if openai_api_key:
        summarizer.set_openai_api_key(openai_api_key)
    
    return summarizer.summarize(abstract)


# =============================================================================
# EXAMPLE USAGE / TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PubMed Connector - Clinical Trial Matching Demo")
    print("=" * 60)
    
    # Example: Search for a trial
    connector = PubMedConnector(email="test@example.com")
    
    # Test NCT ID (a well-known trial)
    test_nct = "NCT02302807"  # KEYNOTE-024 trial
    
    print(f"\nSearching for publications linked to {test_nct}...")
    
    matches = connector.find_publications_for_trial(
        nct_id=test_nct,
        trial_title="Pembrolizumab versus Chemotherapy for PD-L1-Positive Non-Small-Cell Lung Cancer",
        conditions=["Non-Small Cell Lung Cancer", "NSCLC"],
        interventions=["Pembrolizumab", "Keytruda"]
    )
    
    print(f"\nFound {len(matches)} publication(s):\n")
    
    for i, match in enumerate(matches, 1):
        print(f"{i}. [{match.match_method}] Confidence: {match.confidence_score:.2f}")
        print(f"   PMID: {match.article.pmid}")
        print(f"   Title: {match.article.title[:80]}...")
        print(f"   Journal: {match.article.journal}")
        print(f"   Date: {match.article.pub_date}")
        print(f"   NCT IDs in article: {match.article.nct_ids_mentioned}")
        print()
    
    # Test summarization
    if matches:
        print("\n" + "=" * 60)
        print("Outcome Summarization Demo")
        print("=" * 60)
        
        summarizer = OutcomeSummarizer(method='rule_based')
        summary = summarizer.summarize(matches[0].article.abstract)
        
        print(f"\nSummary ({summary['method']}, confidence: {summary['confidence']:.2f}):")
        print(summary['summary'])
        
        # Example of using Gemini (uncomment and add your API key to test)
        print("\n" + "-" * 60)
        print("To use Gemini for better summaries:")
        print("-" * 60)
        print("""
# 1. Install the library:
pip install google-generativeai

# 2. Get your free API key at:
https://aistudio.google.com/app/apikey

# 3. Use it in your code:
import os
os.environ['GEMINI_API_KEY'] = 'your-key-here'

summarizer = OutcomeSummarizer(method='gemini')
summarizer.set_gemini_api_key(os.environ['GEMINI_API_KEY'])
summary = summarizer.summarize(abstract, trial_title="Your Trial Title")
print(summary['summary'])
""")
