from .types import  ComplianceInput
import pandas as pd
from typing import List, Any, Dict, Optional
from loguru import logger
from datetime import timedelta, datetime
from collections import defaultdict
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

    def _parse_date_value(self, value: Any) -> datetime:
        """Parse a date value that could be datetime, Timestamp, or string"""
        if value is None:
            return datetime.min
        if isinstance(value, datetime):
            return value
        if hasattr(value, 'to_pydatetime'):  # pd.Timestamp
            return value.to_pydatetime()
        if isinstance(value, str) and value:
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                return datetime.min
        return datetime.min


    def _verify_row(self, row: dict, calls: List[Any], category: str, manifest_type: str, config: dict) -> Dict[str, str]:
        """Verify compliance for a single row"""
        is_compliant = self.STATUS_CONFORM
        is_dispatched = self.STATUS_CONFORM
        is_branch_calls = self.STATUS_CONFORM
        comments = []
 
        count = row.get('nbr_tentatives_appel', 0)
        date_commande = row.get('date_commande', '')
        date_commande = self._parse_date_value(date_commande)

        if manifest_type == self.MANIFEST_SAV:
            category = row.get('categorie', 'ADSL')

        # 1. Basic check: if no attempts recorded
        if count == 0:
            comments.append("Aucun appel trouvé")
            return {
                'conformite_intervalle': '',
                'appels_branch': '',
                'conformite_IAM': self.STATUS_NON_CONFORM,
                'commentaire': ", ".join(comments)
            }

        # Sort calls by start time
        sorted_calls = sorted(calls, key=self._parse_start_time)

        # 2. Branch Verification
        target_branch = config.get('branche', {}).get(manifest_type, {}).get(category)
        
        if target_branch is None:
            # Configuration missing for this manifest_type/category - skip branch check
            pass
        else:
            for call in sorted_calls:
                if call.get('branch') != target_branch:
                    is_compliant = self.STATUS_NON_CONFORM
                    is_branch_calls = self.STATUS_NON_CONFORM
                    comments.append("Branche non conforme, ")
                    break

        # 3. Client Unreachable Logic
        status = str(row.get('motif_suspension', '')).strip()
        if status.lower() == self.CLIENT_UNREACHABLE:
            # Check 3a: Minimum number of attempts
            if count < self.MIN_ATTEMPTS:
                # Only check call time if we have calls to check
                if sorted_calls:
                    call_time = self._parse_start_time(sorted_calls[-1])
                    # Last call must be after 4pm (16:00) to be compliant
                    if date_commande.hour > 17 and call_time.hour > 17 and date_commande.date() == call_time.date():
                        is_compliant = self.STATUS_CONFORM
                        comments.append("Dernier appel effectué apres 17h")
                    else:
                        is_compliant = self.STATUS_NON_CONFORM
                        comments.append(f"Moins de {self.MIN_ATTEMPTS} appels trouvés")
                else:
                    # No calls available to check
                    is_compliant = self.STATUS_NON_CONFORM
                    comments.append(f"Moins de {self.MIN_ATTEMPTS} appels trouvés")

            # Check 3b: Time gap between consecutive calls
            if len(sorted_calls) >= 3:
                for i in range(len(sorted_calls) - 1):
                    t1 = self._parse_start_time(sorted_calls[i])
                    t2 = self._parse_start_time(sorted_calls[i+1])
                    diff = t2 - t1
                    
                    if diff.total_seconds() < self.MIN_GAP_SECONDS:
                        is_dispatched = self.STATUS_NON_CONFORM
                        is_compliant = self.STATUS_NON_CONFORM
                        hours_diff = diff.total_seconds() / 3600
                        comments.append(f" Le temps entre les appels {i+1} et {i+2} est de {hours_diff:.2f} heures, moins de 2 heures")     
                        break

        return {
            'conformite_intervalle': is_dispatched,
            'appels_branch': is_branch_calls,
            'conformite_IAM': is_compliant,
            'commentaire': ", ".join(comments)
        }

    def verify_compliance(self, df_dict: List[dict], calls_metadata: List[List[Any]], category: str, manifest_type: str, config: dict) -> pd.DataFrame:
        """
        Verify the compliance of the calls.
        - Return the compliance dataframe   
        """
        logger.info(f'df_dict shape: {len(df_dict)}, calls_metadata shape: {len(calls_metadata)}')
        
        for i, row in enumerate(df_dict):
            calls = calls_metadata[i] if i < len(calls_metadata) else []
            result = self._verify_row(row, calls, category, manifest_type, config)
            df_dict[i].update(result)
        return df_dict

    # Default values for beep analysis
    DEFAULT_BEEP_COUNT = 100
    DEFAULT_HIGH_BEEPS = 100
    MOTIF_INJOIGNABLE = 'injoignable'

    def verify_compliance_batch(
        self, 
        df_dict: List[dict], 
        results: List[ComplianceInput]
    ) -> List[dict]:
        """
        Verify the compliance of the calls in batch.
        
        Logic:
        - Group results by numero_commande
        - For each numero_commande with results:
            - If motif_suspension == 'injoignable': find minimum beep_count and 
              check if any result has a different motif_suspension
            - Otherwise: use the first result's values
        - Return the updated list of dicts with compliance info
        
        Args:
            df_dict: List of dictionaries representing rows
            results: List of result objects with numero_commande, beep_count, motif_suspension
            category: Category for compliance check (currently unused)
            manifest_type: Type of manifest (currently unused)
            config: Configuration dict (currently unused)
            
        Returns:
            List[dict]: Updated list with compliance fields added
        """
        # Group results by numero_commande
        results_by_commande: Dict[str, List[Any]] = defaultdict(list)
        for result in results:
            numero_commande = result.numero_commande
            results_by_commande[numero_commande].append(result)
        
        logger.info(f'Processing {len(results_by_commande)} unique commandes from {len(results)} total results')
        
        # Create lookup for df_dict rows by numero_commande
        df_dict_by_commande: Dict[str, dict] = {}
        for row in df_dict:
            numero_commande = row.get('numero_commande')
            if numero_commande:
                df_dict_by_commande[numero_commande] = row
        
        # Process each numero_commande that has results
        for numero_commande, commande_results in results_by_commande.items():
            # Skip if no results or if numero_commande not in our lookup
            if not commande_results or numero_commande not in df_dict_by_commande:
                logger.warning(f'Skipping numero_commande {numero_commande}: no results or not found in df_dict')
                continue
            
            row_data = df_dict_by_commande[numero_commande]
            is_compliant = row_data.get('conformite_IAM')
            commentaire = row_data.get('commentaire') or ''
            current_motif = row_data.get('motif_suspension')
            
            # Initialize with defaults
            beep_count = self.DEFAULT_BEEP_COUNT
            high_beeps = self.DEFAULT_HIGH_BEEPS
            classification_modele = ''
            behavior = ''
            
            if current_motif == self.MOTIF_INJOIGNABLE:
                # For 'injoignable' cases: find minimum beep_count 
                # and check if any result has a different motif
                for commande_res in commande_results:
                    high_beeps = commande_res.high_beeps
                    beep_count = commande_res.beep_count
                    
                    if commande_res.classification.status == 'silence' :
                        if commande_res.beep_count < 5 and commande_res.high_beeps < 1:
                            is_compliant = self.STATUS_NON_CONFORM
                            commentaire += f"Moins de 5 beeps trouvés)"
                        break

                    if commande_res.classification.status != self.MOTIF_INJOIGNABLE:
                        is_compliant = self.STATUS_NON_CONFORM
                        commentaire += f"Classification non conforme: {commande_res.classification.status}"


            else:
                # For non-'injoignable' cases: use values from the first result
                first_result = commande_results[0]
                beep_count = getattr(first_result, 'beep_count', 0)
                high_beeps = getattr(first_result, 'high_beeps', 0)
                classification_data = getattr(first_result, 'classification', None)
                
                # ClassificationResult is a dataclass, not a dict - access attributes directly
                if classification_data:
                    classification_modele = getattr(classification_data, 'status', '')
                    behavior = getattr(classification_data, 'behavior', '')
                else:
                    classification_modele = ''
                    behavior = ''
                    
                if classification_modele != current_motif:
                    is_compliant = self.STATUS_NON_CONFORM
                    commentaire += f"Classification non conforme: {classification_modele}"
            
            # Update the row with computed values
            row_data.update({
                'beep_count': beep_count,
                'high_beeps': high_beeps,
                'classification_modele': 'silence' if current_motif == self.MOTIF_INJOIGNABLE else classification_modele,
                'qualite_communication': behavior,
                'conformite_IAM': is_compliant,
                'commentaire': commentaire
            })
           

        # Convert df_dict_by_commande back to list preserving original order
        updated_df_list = []
        for row in df_dict:
            numero_commande = row.get('numero_commande')
            if numero_commande and numero_commande in df_dict_by_commande:
                # Ensure proper Python types for the updated row
                updated_row = df_dict_by_commande[numero_commande]
                updated_row['beep_count'] = int(updated_row['beep_count']) if updated_row.get('beep_count') is not None else None
                updated_row['high_beeps'] = int(updated_row['high_beeps']) if updated_row.get('high_beeps') is not None else None
                updated_df_list.append(updated_row)
            else:
                # Ensure rows without results have proper default values for required fields
                row['beep_count'] = None
                row['high_beeps'] = None
                row['classification_modele'] = row.get('classification_modele', '')
                row['qualite_communication'] = row.get('qualite_communication', '')
                updated_df_list.append(row)
                
        logger.info(f'Compliance verification completed for {len(updated_df_list)} commandes')
        return updated_df_list
