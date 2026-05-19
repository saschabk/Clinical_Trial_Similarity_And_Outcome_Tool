"""
Clinical Trial Similarity Engine (Enhanced Version)
Added additional text fields for improved similarity matching.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Any, Optional
import pickle


class TrialSimilarityEngine:
    """Find similar clinical trials with drug/disease filtering."""
    
    def __init__(self):
        self.trials_df = None
        self.design_features = None
        self.text_features = None
        self.outcome_features = None
        self.scaler = StandardScaler()
        # Increased max_features to capture more vocabulary from additional text
        self.text_vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        self.outcome_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        self.trial_index = {}
        
    def build_index(self, studies: List[Dict[str, Any]]):
        """Build searchable index from studies."""
        print(f"Building index from {len(studies)} trials...")
        
        trials_data = []
        for idx, study in enumerate(studies):
            try:
                features = self._extract_features(study)
                trials_data.append(features)
                
                if (idx + 1) % 1000 == 0:
                    print(f"  Processed {idx + 1:,} trials...")
            except:
                continue
        
        self.trials_df = pd.DataFrame(trials_data)
        self.trial_index = {nct_id: idx for idx, nct_id in enumerate(self.trials_df['nct_id'])}
        
        self._encode_features()
        
        print(f"✓ Indexed {len(self.trials_df):,} trials")
        
    def _extract_features(self, study: Dict[str, Any]) -> Dict[str, Any]:
        """Extract all features from a single study."""
        p = study.get('protocolSection', {})
        
        # Shortcuts to nested dicts
        ident = p.get('identificationModule', {})
        status = p.get('statusModule', {})
        design = p.get('designModule', {})
        design_info = design.get('designInfo', {})
        arms = p.get('armsInterventionsModule', {})
        outcomes = p.get('outcomesModule', {})
        eligibility = p.get('eligibilityModule', {})
        conditions = p.get('conditionsModule', {})
        description = p.get('descriptionModule', {})
        sponsor = p.get('sponsorCollaboratorsModule', {})
        
        # Extract lists
        interventions = arms.get('interventions', [])
        arm_groups = arms.get('armGroups', [])
        primary_outcomes = outcomes.get('primaryOutcomes', [])
        secondary_outcomes = outcomes.get('secondaryOutcomes', [])
        condition_list = conditions.get('conditions', [])
        keyword_list = conditions.get('keywords', [])  # NEW: Keywords
        intervention_names = [i.get('name', '') for i in interventions]
        
        # Build outcome text (includes timeframes now)
        outcome_text = ' '.join([
            o.get('measure', '') + ' ' + o.get('description', '') + ' ' + o.get('timeFrame', '')
            for o in primary_outcomes + secondary_outcomes
        ])
        
        # NEW: Extract intervention descriptions
        intervention_descriptions = ' '.join([
            i.get('description', '') for i in interventions
        ])
        
        # NEW: Extract arm/group descriptions
        arm_descriptions = ' '.join([
            arm.get('description', '') + ' ' + arm.get('label', '')
            for arm in arm_groups
        ])
        
        # NEW: Extract eligibility criteria text
        eligibility_criteria = eligibility.get('eligibilityCriteria', '')
        
        return {
            'nct_id': ident.get('nctId', 'UNKNOWN'),
            'brief_title': ident.get('briefTitle', ''),
            'official_title': ident.get('officialTitle', ''),
            'acronym': ident.get('acronym', ''),  # NEW: Study acronym (e.g., "FLAURA", "KEYNOTE-001")
            'overall_status': status.get('overallStatus', 'UNKNOWN'),
            'study_type': design.get('studyType', 'UNKNOWN'),
            'phase': ','.join(design.get('phases', [])) or 'NA',
            'allocation': design_info.get('allocation', 'NA'),
            'intervention_model': design_info.get('interventionModel', 'NA'),
            'primary_purpose': design_info.get('primaryPurpose', 'NA'),
            'masking': design_info.get('maskingInfo', {}).get('masking', 'NA'),
            'observational_model': design_info.get('observationalModel', 'NA'),
            'time_perspective': design_info.get('timePerspective', 'NA'),
            'enrollment_count': design.get('enrollmentInfo', {}).get('count', 0),
            'num_arms': len(arm_groups),
            'num_interventions': len(interventions),
            'num_primary_outcomes': len(primary_outcomes),
            'num_secondary_outcomes': len(secondary_outcomes),
            'num_locations': len(p.get('contactsLocationsModule', {}).get('locations', [])),
            'sex': eligibility.get('sex', 'ALL'),
            'healthy_volunteers': eligibility.get('healthyVolunteers', False),
            'min_age': eligibility.get('minimumAge', ''),  # NEW
            'max_age': eligibility.get('maximumAge', ''),  # NEW
            'conditions': ', '.join(condition_list),
            'conditions_list': condition_list,
            'keywords': ', '.join(keyword_list),  # NEW: Keywords field
            'keywords_list': keyword_list,  # NEW
            'intervention_names': ', '.join(intervention_names),
            'intervention_names_list': intervention_names,
            'intervention_types': ','.join([i.get('type', '') for i in interventions]),
            'intervention_descriptions': intervention_descriptions,  # NEW
            'arm_descriptions': arm_descriptions,  # NEW
            'brief_summary': description.get('briefSummary', ''),
            'detailed_description': description.get('detailedDescription', ''),  # NEW
            'eligibility_criteria': eligibility_criteria,  # NEW
            'sponsor_name': sponsor.get('leadSponsor', {}).get('name', ''),
            'sponsor_class': sponsor.get('leadSponsor', {}).get('class', ''),
            'primary_outcome_measures': ' | '.join([o.get('measure', '') for o in primary_outcomes]),
            'secondary_outcome_measures': ' | '.join([o.get('measure', '') for o in secondary_outcomes]),
            'outcome_text_combined': outcome_text,
        }
    
    def _encode_features(self):
        """Encode all features into numerical matrices."""
        # Categorical features
        cat_cols = ['study_type', 'phase', 'allocation', 'intervention_model',
                    'primary_purpose', 'masking', 'observational_model',
                    'time_perspective', 'sex', 'overall_status']
        
        categorical = pd.get_dummies(self.trials_df[cat_cols], prefix=cat_cols)
        
        # Numerical features
        num_cols = ['enrollment_count', 'num_arms', 'num_interventions',
                    'num_primary_outcomes', 'num_secondary_outcomes', 'num_locations']
        
        numerical = self.scaler.fit_transform(self.trials_df[num_cols].fillna(0))
        
        # Boolean
        boolean = self.trials_df['healthy_volunteers'].astype(int).values.reshape(-1, 1)
        
        # Combine design features
        self.design_features = np.hstack([categorical.values, numerical, boolean])
        
        # ENHANCED Text features - now includes many more fields
        text_data = (
            # Core identification
            self.trials_df['brief_title'].fillna('') + ' ' +
            self.trials_df['official_title'].fillna('') + ' ' +
            self.trials_df['acronym'].fillna('') + ' ' +
            # Disease/condition info
            self.trials_df['conditions'].fillna('') + ' ' +
            self.trials_df['keywords'].fillna('') + ' ' +
            # Intervention info
            self.trials_df['intervention_names'].fillna('') + ' ' +
            self.trials_df['intervention_descriptions'].fillna('') + ' ' +
            # Study descriptions
            self.trials_df['brief_summary'].fillna('') + ' ' +
            self.trials_df['detailed_description'].fillna('') + ' ' +
            # Arm/group descriptions
            self.trials_df['arm_descriptions'].fillna('')
        )
        self.text_features = self.text_vectorizer.fit_transform(text_data).toarray()
        
        # Outcome features (kept separate for weighted scoring)
        self.outcome_features = self.outcome_vectorizer.fit_transform(
            self.trials_df['outcome_text_combined'].fillna('')
        ).toarray()
    
    def find_similar_by_nct(self, 
                           nct_id: str, 
                           top_k: int = 10,
                           weights: Optional[Dict[str, float]] = None,
                           same_drug: bool = False,
                           same_disease: bool = False) -> pd.DataFrame:
        """
        Find similar trials.
        
        Args:
            nct_id: Query trial NCT ID
            top_k: Number of results
            weights: {'design': 0.5, 'text': 0.3, 'outcomes': 0.2}
            same_drug: Filter to same drug/intervention
            same_disease: Filter to same disease/condition
        """
        if nct_id not in self.trial_index:
            raise ValueError(f"NCT ID {nct_id} not found")
        
        if weights is None:
            weights = {'design': 0.5, 'text': 0.2, 'outcomes': 0.3}
        
        query_idx = self.trial_index[nct_id]
        query_trial = self.trials_df.iloc[query_idx]
        
        # Calculate similarities
        design_sim = cosine_similarity(
            self.design_features[query_idx].reshape(1, -1),
            self.design_features
        )[0]
        
        text_sim = cosine_similarity(
            self.text_features[query_idx].reshape(1, -1),
            self.text_features
        )[0]
        
        outcome_sim = cosine_similarity(
            self.outcome_features[query_idx].reshape(1, -1),
            self.outcome_features
        )[0]
        
        # Combined similarity
        combined = (
            weights['design'] * design_sim +
            weights['text'] * text_sim +
            weights['outcomes'] * outcome_sim
        )
        
        # Exclude query trial
        combined[query_idx] = -1
        
        # Apply filters
        if same_drug or same_disease:
            query_drugs = set(query_trial['intervention_names_list'])
            query_diseases = set(query_trial['conditions_list'])
            
            for idx in range(len(self.trials_df)):
                if idx == query_idx:
                    continue
                
                trial = self.trials_df.iloc[idx]
                
                if same_drug:
                    trial_drugs = set(trial['intervention_names_list'])
                    if not query_drugs.intersection(trial_drugs):
                        combined[idx] = -1
                
                if same_disease:
                    trial_diseases = set(trial['conditions_list'])
                    if not query_diseases.intersection(trial_diseases):
                        combined[idx] = -1
        
        # Get top results
        top_indices = np.argsort(combined)[::-1][:top_k * 2]
        
        results = []
        for idx in top_indices:
            if combined[idx] < 0 or len(results) >= top_k:
                continue
            
            trial = self.trials_df.iloc[idx]
            results.append({
                'nct_id': trial['nct_id'],
                'brief_title': trial['brief_title'],
                'study_type': trial['study_type'],
                'phase': trial['phase'],
                'conditions': trial['conditions'],
                'interventions': trial['intervention_names'],
                'primary_outcomes': trial['primary_outcome_measures'],
                'overall_similarity': combined[idx],
                'design_similarity': design_sim[idx],
                'text_similarity': text_sim[idx],
                'outcome_similarity': outcome_sim[idx],
                'enrollment': trial['enrollment_count'],
                'num_arms': trial['num_arms'],
                'sponsor': trial['sponsor_name']
            })
        
        return pd.DataFrame(results)
    
    def save_index(self, filepath: str):
        """Save index to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'trials_df': self.trials_df,
                'design_features': self.design_features,
                'text_features': self.text_features,
                'outcome_features': self.outcome_features,
                'scaler': self.scaler,
                'text_vectorizer': self.text_vectorizer,
                'outcome_vectorizer': self.outcome_vectorizer,
                'trial_index': self.trial_index
            }, f)
        print(f"✓ Saved: {filepath}")
    
    def load_index(self, filepath: str):
        """Load index from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.trials_df = data['trials_df']
        self.design_features = data['design_features']
        self.text_features = data['text_features']
        self.outcome_features = data['outcome_features']
        self.scaler = data['scaler']
        self.text_vectorizer = data['text_vectorizer']
        self.outcome_vectorizer = data['outcome_vectorizer']
        self.trial_index = data['trial_index']
        
        print(f"✓ Loaded: {len(self.trials_df):,} trials")