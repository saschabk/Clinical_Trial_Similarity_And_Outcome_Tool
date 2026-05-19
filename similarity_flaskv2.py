"""
Flask Backend - Enhanced with PubMed Integration

This version adds endpoints for:
- Finding linked publications for trials
- Outcome summarization
- Batch publication matching for similar trials
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys

# Import the similarity engine
from similarity import TrialSimilarityEngine

# Import PubMed connector
from pubmed_connector import (
    PubMedConnector, 
    OutcomeSummarizer,
    match_trial_to_publications
)

app = Flask(__name__)
CORS(app)

# Global instances
engine = None
pubmed = None
summarizer = None

# Configuration
INDEX_FILE = 'trial_similarity_index_complete.pkl'  

# Optional
PUBMED_EMAIL = '123@gmail.com'
PUBMED_API_KEY = '123'


def initialize_engine():
    """Load the similarity index and initialize PubMed connector."""
    global engine, pubmed, summarizer
    
    # Initialize similarity engine
    engine = TrialSimilarityEngine()
    
    if os.path.exists(INDEX_FILE):
        engine.load_index(INDEX_FILE)
        print(f"✓ Loaded {len(engine.trials_df):,} trials")
    else:
        print(f"✗ Index not found: {INDEX_FILE}")
    
    # Initialize PubMed connector
    pubmed = PubMedConnector(email=PUBMED_EMAIL, api_key=PUBMED_API_KEY)
    print(f"✓ PubMed connector initialized (email: {PUBMED_EMAIL})")
    
    # Initialize summarizer
    summarizer = OutcomeSummarizer(method='rule_based')
    print("✓ Outcome summarizer initialized (rule-based)")


# =============================================================================
# EXISTING ENDPOINTS (unchanged)
# =============================================================================

@app.route('/health', methods=['GET'])
def health():
    """Health check."""
    return jsonify({
        'status': 'healthy',
        'trials': len(engine.trials_df) if engine and engine.trials_df is not None else 0,
        'pubmed_connected': pubmed is not None
    })


@app.route('/api/search', methods=['POST'])
def search():
    """
    Search for similar trials.
    
    POST body:
    {
        "nct_id": "NCT04280705",
        "top_k": 10,
        "weights": {"design": 0.5, "text": 0.3, "outcomes": 0.2},
        "same_drug": false,
        "same_disease": false,
        "include_publications": false  // NEW: optionally fetch publications
    }
    """
    try:
        data = request.json
        nct_id = data.get('nct_id', '').strip().upper()
        
        if not nct_id:
            return jsonify({'error': 'NCT ID required'}), 400
        
        if nct_id not in engine.trial_index:
            return jsonify({'error': f'Trial {nct_id} not found'}), 404
        
        # Get query trial info
        query_idx = engine.trial_index[nct_id]
        q = engine.trials_df.iloc[query_idx]
        
        query_trial = {
            'nct_id': q['nct_id'],
            'brief_title': q['brief_title'],
            'study_type': q['study_type'],
            'phase': q['phase'],
            'conditions': q['conditions'],
            'interventions': q['intervention_names'],
            'primary_outcomes': q['primary_outcome_measures'],
            'enrollment': int(q['enrollment_count']),
            'sponsor': q['sponsor_name'],
            'status': q['overall_status']
        }
        
        # Search for similar
        similar = engine.find_similar_by_nct(
            nct_id,
            top_k=data.get('top_k', 10),
            weights=data.get('weights'),
            same_drug=data.get('same_drug', False),
            same_disease=data.get('same_disease', False)
        )
        
        similar_trials = similar.to_dict('records')
        
        # Optionally fetch publications for each similar trial
        if data.get('include_publications', False):
            for trial in similar_trials:
                try:
                    # Parse conditions and interventions from string
                    conditions = trial.get('conditions', '').split(', ') if trial.get('conditions') else []
                    interventions = trial.get('interventions', '').split(', ') if trial.get('interventions') else []
                    
                    matches = pubmed.find_publications_for_trial(
                        nct_id=trial['nct_id'],
                        trial_title=trial.get('brief_title'),
                        conditions=conditions[:3],
                        interventions=interventions[:3]
                    )
                    
                    # Add top match to trial data
                    if matches:
                        top_match = matches[0]
                        trial['publication'] = {
                            'pmid': top_match.article.pmid,
                            'title': top_match.article.title,
                            'journal': top_match.article.journal,
                            'pub_date': top_match.article.pub_date,
                            'match_confidence': top_match.confidence_score,
                            'match_method': top_match.match_method,
                            'pubmed_url': f"https://pubmed.ncbi.nlm.nih.gov/{top_match.article.pmid}/"
                        }
                    else:
                        trial['publication'] = None
                        
                except Exception as e:
                    print(f"Error fetching publication for {trial['nct_id']}: {e}")
                    trial['publication'] = None
        
        return jsonify({
            'query_trial': query_trial,
            'similar_trials': similar_trials,
            'total_found': len(similar_trials),
            'filters_applied': {
                'same_drug': data.get('same_drug', False),
                'same_disease': data.get('same_disease', False)
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/trial/<nct_id>', methods=['GET'])
def get_trial(nct_id):
    """Get trial info."""
    try:
        nct_id = nct_id.strip().upper()
        
        if nct_id not in engine.trial_index:
            return jsonify({'error': f'Not found: {nct_id}'}), 404
        
        trial = engine.trials_df.iloc[engine.trial_index[nct_id]]
        
        return jsonify({
            'nct_id': trial['nct_id'],
            'brief_title': trial['brief_title'],
            'study_type': trial['study_type'],
            'phase': trial['phase'],
            'conditions': trial['conditions'],
            'interventions': trial['intervention_names'],
            'enrollment': int(trial['enrollment_count']),
            'sponsor': trial['sponsor_name']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def stats():
    """Database statistics."""
    try:
        if engine.trials_df is None:
            return jsonify({'error': 'Index not loaded'}), 500
        
        df = engine.trials_df
        
        return jsonify({
            'total_trials': len(df),
            'study_types': df['study_type'].value_counts().to_dict(),
            'phases': df['phase'].value_counts().head(10).to_dict(),
            'status': df['overall_status'].value_counts().to_dict()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =============================================================================
# NEW PUBMED ENDPOINTS
# =============================================================================

@app.route('/api/publications/<nct_id>', methods=['GET'])
def get_publications(nct_id):
    """
    Find PubMed publications linked to a clinical trial.
    
    Query params:
        - max_results: Maximum number of publications (default: 5)
        - include_outcomes: Whether to include outcome extraction (default: true)
    
    Returns:
        List of matched publications with confidence scores
    """
    try:
        nct_id = nct_id.strip().upper()
        max_results = int(request.args.get('max_results', 5))
        include_outcomes = request.args.get('include_outcomes', 'true').lower() == 'true'
        
        # Get trial info from our index if available
        trial_title = None
        conditions = []
        interventions = []
        
        if engine and nct_id in engine.trial_index:
            trial = engine.trials_df.iloc[engine.trial_index[nct_id]]
            trial_title = trial['brief_title']
            conditions = trial['conditions_list'] if isinstance(trial['conditions_list'], list) else []
            interventions = trial['intervention_names_list'] if isinstance(trial['intervention_names_list'], list) else []
        
        # Find publications
        matches = pubmed.find_publications_for_trial(
            nct_id=nct_id,
            trial_title=trial_title,
            conditions=conditions[:3],
            interventions=interventions[:3]
        )
        
        # Limit results
        matches = matches[:max_results]
        
        # Format results
        results = []
        for match in matches:
            result = {
                'pmid': match.article.pmid,
                'title': match.article.title,
                'authors': match.article.authors[:5],  # First 5 authors
                'journal': match.article.journal,
                'pub_date': match.article.pub_date,
                'doi': match.article.doi,
                'pmc_id': match.article.pmc_id,
                'abstract': match.article.abstract,
                'publication_types': match.article.publication_types,
                'mesh_terms': match.article.mesh_terms[:10],  # First 10
                'pubmed_url': f"https://pubmed.ncbi.nlm.nih.gov/{match.article.pmid}/",
                'full_text_url': f"https://www.ncbi.nlm.nih.gov/pmc/articles/{match.article.pmc_id}/" if match.article.pmc_id else None,
                'match_method': match.match_method,
                'match_confidence': match.confidence_score,
                'match_details': match.match_details
            }
            
            # Extract outcomes if requested
            if include_outcomes and match.article.abstract:
                outcomes = pubmed.extract_outcomes_from_abstract(match.article.abstract)
                result['outcomes_extracted'] = outcomes
                
                # Also generate summary
                summary = summarizer.summarize(match.article.abstract)
                result['outcome_summary'] = summary
            
            results.append(result)
        
        return jsonify({
            'nct_id': nct_id,
            'trial_title': trial_title,
            'publications_found': len(results),
            'publications': results
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/publications/batch', methods=['POST'])
def get_publications_batch():
    """
    Find publications for multiple trials at once.
    
    POST body:
    {
        "nct_ids": ["NCT01234567", "NCT02345678", ...],
        "max_per_trial": 1  // Max publications per trial
    }
    
    Returns:
        Dictionary mapping NCT IDs to their publications
    """
    try:
        data = request.json
        nct_ids = data.get('nct_ids', [])
        max_per_trial = data.get('max_per_trial', 1)
        
        if not nct_ids:
            return jsonify({'error': 'nct_ids required'}), 400
        
        if len(nct_ids) > 20:
            return jsonify({'error': 'Maximum 20 trials per batch'}), 400
        
        results = {}
        
        for nct_id in nct_ids:
            nct_id = nct_id.strip().upper()
            
            try:
                # Get trial info
                trial_title = None
                conditions = []
                interventions = []
                
                if engine and nct_id in engine.trial_index:
                    trial = engine.trials_df.iloc[engine.trial_index[nct_id]]
                    trial_title = trial['brief_title']
                    conditions = trial['conditions_list'] if isinstance(trial['conditions_list'], list) else []
                    interventions = trial['intervention_names_list'] if isinstance(trial['intervention_names_list'], list) else []
                
                # Find publications
                matches = pubmed.find_publications_for_trial(
                    nct_id=nct_id,
                    trial_title=trial_title,
                    conditions=conditions[:2],
                    interventions=interventions[:2]
                )
                
                if matches:
                    top_match = matches[0]
                    
                    # Generate summary
                    summary = None
                    if top_match.article.abstract:
                        summary = summarizer.summarize(top_match.article.abstract)
                    
                    results[nct_id] = {
                        'found': True,
                        'pmid': top_match.article.pmid,
                        'title': top_match.article.title,
                        'journal': top_match.article.journal,
                        'pub_date': top_match.article.pub_date,
                        'match_confidence': top_match.confidence_score,
                        'match_method': top_match.match_method,
                        'pubmed_url': f"https://pubmed.ncbi.nlm.nih.gov/{top_match.article.pmid}/",
                        'outcome_summary': summary
                    }
                else:
                    results[nct_id] = {
                        'found': False,
                        'message': 'No publication found'
                    }
                    
            except Exception as e:
                results[nct_id] = {
                    'found': False,
                    'error': str(e)
                }
        
        return jsonify({
            'results': results,
            'total_requested': len(nct_ids),
            'total_found': sum(1 for r in results.values() if r.get('found', False))
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/summarize', methods=['POST'])
def summarize_outcomes():
    """
    Summarize outcomes from text.
    
    POST body:
    {
        "abstract": "Study abstract text...",
        "method": "rule_based",  // or "transformer" or "api"
        "max_length": 150
    }
    """
    try:
        data = request.json
        abstract = data.get('abstract', '')
        method = data.get('method', 'rule_based')
        max_length = data.get('max_length', 150)
        
        if not abstract:
            return jsonify({'error': 'abstract required'}), 400
        
        # Use the configured summarizer or create a new one
        if method == summarizer.method:
            summary = summarizer.summarize(abstract, max_length=max_length)
        else:
            temp_summarizer = OutcomeSummarizer(method=method)
            summary = temp_summarizer.summarize(abstract, max_length=max_length)
        
        return jsonify(summary)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/search-pubmed', methods=['POST'])
def search_pubmed():
    """
    Direct PubMed search.
    
    POST body:
    {
        "query": "search query",
        "max_results": 10
    }
    """
    try:
        data = request.json
        query = data.get('query', '').strip()
        max_results = min(data.get('max_results', 10), 50)
        
        if not query:
            return jsonify({'error': 'query required'}), 400
        
        # Use ESearch
        import requests
        url = f"{pubmed.BASE_URL}/esearch.fcgi"
        params = pubmed._build_params(
            db='pubmed',
            term=query,
            retmax=max_results,
            retmode='json'
        )
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        search_data = response.json()
        
        pmids = search_data.get('esearchresult', {}).get('idlist', [])
        
        if not pmids:
            return jsonify({
                'query': query,
                'total_found': 0,
                'results': []
            })
        
        # Fetch article details
        articles = pubmed.fetch_articles(pmids)
        
        results = []
        for article in articles:
            results.append({
                'pmid': article.pmid,
                'title': article.title,
                'authors': article.authors[:3],
                'journal': article.journal,
                'pub_date': article.pub_date,
                'abstract_preview': article.abstract[:300] + '...' if len(article.abstract) > 300 else article.abstract,
                'nct_ids': article.nct_ids_mentioned,
                'pubmed_url': f"https://pubmed.ncbi.nlm.nih.gov/{article.pmid}/"
            })
        
        return jsonify({
            'query': query,
            'total_found': int(search_data.get('esearchresult', {}).get('count', 0)),
            'returned': len(results),
            'results': results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =============================================================================
# COMBINED SEARCH ENDPOINT (Similar trials + Publications)
# =============================================================================

@app.route('/api/full-search', methods=['POST'])
def full_search():
    """
    Complete search: find similar trials AND their publications.
    
    POST body:
    {
        "nct_id": "NCT04280705",
        "top_k": 10,
        "weights": {"design": 0.5, "text": 0.3, "outcomes": 0.2},
        "same_drug": false,
        "same_disease": false,
        "fetch_publications": true
    }
    
    Returns similar trials with publication info and outcome summaries.
    """
    try:
        data = request.json
        nct_id = data.get('nct_id', '').strip().upper()
        fetch_publications = data.get('fetch_publications', True)
        
        if not nct_id:
            return jsonify({'error': 'NCT ID required'}), 400
        
        if nct_id not in engine.trial_index:
            return jsonify({'error': f'Trial {nct_id} not found'}), 404
        
        # Get query trial info
        query_idx = engine.trial_index[nct_id]
        q = engine.trials_df.iloc[query_idx]
        
        query_trial = {
            'nct_id': q['nct_id'],
            'brief_title': q['brief_title'],
            'study_type': q['study_type'],
            'phase': q['phase'],
            'conditions': q['conditions'],
            'interventions': q['intervention_names'],
            'primary_outcomes': q['primary_outcome_measures'],
            'enrollment': int(q['enrollment_count']),
            'sponsor': q['sponsor_name'],
            'status': q['overall_status']
        }
        
        # Also fetch publication for query trial
        if fetch_publications:
            try:
                conditions = q['conditions_list'] if isinstance(q['conditions_list'], list) else []
                interventions = q['intervention_names_list'] if isinstance(q['intervention_names_list'], list) else []
                
                matches = pubmed.find_publications_for_trial(
                    nct_id=nct_id,
                    trial_title=q['brief_title'],
                    conditions=conditions[:3],
                    interventions=interventions[:3]
                )
                
                if matches:
                    top = matches[0]
                    query_trial['publication'] = {
                        'pmid': top.article.pmid,
                        'title': top.article.title,
                        'journal': top.article.journal,
                        'pub_date': top.article.pub_date,
                        'abstract': top.article.abstract,
                        'match_confidence': top.confidence_score,
                        'pubmed_url': f"https://pubmed.ncbi.nlm.nih.gov/{top.article.pmid}/"
                    }
                    
                    # Summarize outcomes
                    if top.article.abstract:
                        summary = summarizer.summarize(top.article.abstract)
                        query_trial['publication']['outcome_summary'] = summary
            except Exception as e:
                print(f"Error fetching publication for query trial: {e}")
                query_trial['publication'] = None
        
        # Find similar trials
        similar = engine.find_similar_by_nct(
            nct_id,
            top_k=data.get('top_k', 10),
            weights=data.get('weights'),
            same_drug=data.get('same_drug', False),
            same_disease=data.get('same_disease', False)
        )
        
        similar_trials = similar.to_dict('records')
        
        # Fetch publications for similar trials
        if fetch_publications:
            for trial in similar_trials:
                try:
                    trial_idx = engine.trial_index.get(trial['nct_id'])
                    if trial_idx is not None:
                        t = engine.trials_df.iloc[trial_idx]
                        conditions = t['conditions_list'] if isinstance(t['conditions_list'], list) else []
                        interventions = t['intervention_names_list'] if isinstance(t['intervention_names_list'], list) else []
                    else:
                        conditions = trial.get('conditions', '').split(', ') if trial.get('conditions') else []
                        interventions = trial.get('interventions', '').split(', ') if trial.get('interventions') else []
                    
                    matches = pubmed.find_publications_for_trial(
                        nct_id=trial['nct_id'],
                        trial_title=trial.get('brief_title'),
                        conditions=conditions[:2],
                        interventions=interventions[:2]
                    )
                    
                    if matches:
                        top = matches[0]
                        trial['publication'] = {
                            'pmid': top.article.pmid,
                            'title': top.article.title,
                            'journal': top.article.journal,
                            'pub_date': top.article.pub_date,
                            'match_confidence': top.confidence_score,
                            'match_method': top.match_method,
                            'pubmed_url': f"https://pubmed.ncbi.nlm.nih.gov/{top.article.pmid}/"
                        }
                        
                        # Summarize outcomes
                        if top.article.abstract:
                            summary = summarizer.summarize(top.article.abstract)
                            trial['publication']['outcome_summary'] = summary
                    else:
                        trial['publication'] = None
                        
                except Exception as e:
                    print(f"Error fetching publication for {trial['nct_id']}: {e}")
                    trial['publication'] = None
        
        return jsonify({
            'query_trial': query_trial,
            'similar_trials': similar_trials,
            'total_found': len(similar_trials),
            'publications_fetched': fetch_publications,
            'filters_applied': {
                'same_drug': data.get('same_drug', False),
                'same_disease': data.get('same_disease', False)
            }
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    initialize_engine()
    print("\n" + "=" * 50)
    print("API Endpoints:")
    print("  GET  /health              - Health check")
    print("  POST /api/search          - Find similar trials")
    print("  GET  /api/trial/<nct_id>  - Get trial info")
    print("  GET  /api/stats           - Database statistics")
    print("  GET  /api/publications/<nct_id> - Find publications")
    print("  POST /api/publications/batch    - Batch publication lookup")
    print("  POST /api/summarize       - Summarize outcomes")
    print("  POST /api/search-pubmed   - Direct PubMed search")
    print("  POST /api/full-search     - Similar trials + publications")
    print("=" * 50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5001)
