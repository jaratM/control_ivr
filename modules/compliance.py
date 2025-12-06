from .types import ComplianceResult
import pandas as pd
from typing import List, Any, Dict, Optional
from loguru import logger
from datetime import timedelta, datetime

class ComplianceVerifier:
    """
    Compliance verifier.
    - Verify the compliance of the calls
    - Return the compliance result
    """
    # Constants
    STATUS_CONFORM = 'Conform'
    STATUS_NON_CONFORM = 'Non conforme'
    CLIENT_UNREACHABLE = 'client injoignable'
    MANIFEST_ACQUISITION = 'ACQUISITION'
    MANIFEST_SAV = 'SAV'
    MIN_ATTEMPTS = 3
    MIN_GAP_SECONDS = 7200 # 2 hours

    def _parse_start_time(self, call: Dict) -> datetime:
        """Helper to parse start_time (handles both datetime objects and ISO strings from serialization)"""
        st = call.get('start_time')
        if st is None:
            return datetime.min
        if isinstance(st, datetime):
            return st
        if isinstance(st, str):
            try:
                return datetime.fromisoformat(st)
            except ValueError:
                return datetime.min
        return datetime.min


    def _verify_row(self, row: pd.Series, calls: List[Any], category: str, manifest_type: str, config: dict) -> Dict[str, str]:
        """Verify compliance for a single row"""
        is_compliant = self.STATUS_CONFORM
        is_dispatched = self.STATUS_CONFORM
        is_branch_calls = self.STATUS_CONFORM
        comments = []

        count = row.get('Nbr_tentatives_appel', 0)
        status = str(row.get('status', '')).strip()
        if manifest_type == self.MANIFEST_SAV:
            category = row.get('categorie', '')

        # 1. Basic check: if no attempts recorded
        if count == 0:
            comments.append("Aucun appel trouvé")
            return {
                'appels_dispatches': is_dispatched,
                'appels_branch': is_branch_calls,
                'compliance': self.STATUS_NON_CONFORM,
                'commentaires': ", ".join(comments)
            }

        # Sort calls by start time
        sorted_calls = sorted(calls, key=self._parse_start_time)

        # 2. Branch Verification
        target_branch = config.get('branche', {}).get(manifest_type, {}).get(category)
        
        if target_branch is None:
             is_compliant = self.STATUS_NON_CONFORM
             is_branch_calls = self.STATUS_NON_CONFORM
             comments.append("Branche non conforme")
        else:
            for call in sorted_calls:
                if call.get('branch') != target_branch:
                    is_compliant = self.STATUS_NON_CONFORM
                    is_branch_calls = self.STATUS_NON_CONFORM
                    comments.append("Branche non conforme")
                    break

        # 3. Client Unreachable Logic
        if status.lower() == self.CLIENT_UNREACHABLE:
            # Check 3a: Minimum number of attempts
            if count < self.MIN_ATTEMPTS:
                is_compliant = self.STATUS_NON_CONFORM
                comments.append(f"Moins de {self.MIN_ATTEMPTS} appels trouvés")

            # Check 3b: Time gap between consecutive calls
            if len(sorted_calls) >= 2:
                for i in range(len(sorted_calls) - 1):
                    t1 = self._parse_start_time(sorted_calls[i])
                    t2 = self._parse_start_time(sorted_calls[i+1])
                    diff = t2 - t1
                    
                    if diff.total_seconds() < self.MIN_GAP_SECONDS:
                        is_dispatched = self.STATUS_NON_CONFORM
                        is_compliant = self.STATUS_NON_CONFORM
                        hours_diff = diff.total_seconds() / 3600
                        comments.append(f"Le temps entre les appels {i+1} et {i+2} est de {hours_diff:.2f} heures, moins de 2 heures")     
                        break

        return {
            'appels_dispatches': is_dispatched,
            'appels_branch': is_branch_calls,
            'compliance': is_compliant,
            'commentaires': ", ".join(comments)
        }

    def verify_compliance(self, df: pd.DataFrame, calls_metadata: List[List[Any]], category: str, manifest_type: str, config: dict) -> pd.DataFrame:
        """
        Verify the compliance of the calls.
        - Return the compliance dataframe   
        """
        logger.info(f'dataframe shape: {df.shape}, calls_metadata shape: {len(calls_metadata)}')
        
        results = []
        for i, (_, row) in enumerate(df.iterrows()):
            calls = calls_metadata[i] if i < len(calls_metadata) else []
            result = self._verify_row(row, calls, category, manifest_type, config)
            results.append(result)
            
        # Bulk assign columns
        df['appels_dispatches'] = [r['appels_dispatches'] for r in results]
        df['appels_branch'] = [r['appels_branch'] for r in results]
        df['compliance'] = [r['compliance'] for r in results]
        df['commentaires'] = [r['commentaires'] for r in results]

        return df

    def verify_compliance_batch(self, df: pd.DataFrame, calls_metadata: List[List[Any]], results: List[Any], category: str, manifest_type: str, config: dict) -> pd.DataFrame:
        """
        Verify the compliance of the calls in batch.
        - Bulk assign the compliance result to the dataframe
        - Return the compliance dataframe
        """
        