from .types import  ComplianceInput
import pandas as pd
from typing import List, Any, Dict, Optional, Tuple
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

        # Default values for beep analysis
    DEFAULT_BEEP_COUNT = 100
    DEFAULT_HIGH_BEEPS = 100
    MOTIF_INJOIGNABLE = 'injoignable'

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



    def _group_results_by_commande(
        self, 
        results: List[ComplianceInput]
    ) -> Dict[str, List[ComplianceInput]]:
        """
        Group compliance results by numero_commande.
        
        Args:
            results: List of ComplianceInput objects to group
            
        Returns:
            Dictionary mapping numero_commande to list of ComplianceInput objects
        """
        results_by_commande: Dict[str, List[ComplianceInput]] = defaultdict(list)
        for result in results:
            results_by_commande[result.numero_commande].append(result)
        return results_by_commande
    
    def _create_commande_lookup(
        self, 
        df_dict: List[dict]
    ) -> Dict[str, dict]:
        """
        Create a lookup dictionary mapping numero_commande to row data.
        
        Args:
            df_dict: List of dictionaries representing DataFrame rows
            
        Returns:
            Dictionary mapping numero_commande to row dictionary
        """
        df_dict_by_commande: Dict[str, dict] = {}
        for row in df_dict:
            numero_commande = row.get('numero_commande')
            if numero_commande:
                df_dict_by_commande[numero_commande] = row
        return df_dict_by_commande
    
    def _process_injoignable_commande(
        self,
        commande_results: List[ComplianceInput],
        row_data: dict
    ) -> Tuple[int, int, str, str, str, str]:
        """
        Process compliance verification for 'injoignable' motif cases.
        
        For injoignable cases, the method:
        - Iterates through all results to find beep counts
        - Checks if any result has 'silence' status with insufficient beeps (< 5 beeps and < 1 high beep)
        - Verifies that all classifications match the 'injoignable' motif
        
        Args:
            commande_results: List of ComplianceInput results for this commande
            row_data: Dictionary containing the row data to update
            
        Returns:
            Tuple of (beep_count, high_beeps, classification_modele, behavior, 
                     is_compliant, commentaire)
        """
        is_compliant = row_data.get('conformite_IAM', self.STATUS_CONFORM)
        commentaire = row_data.get('commentaire', '') or ''
        beep_count = self.DEFAULT_BEEP_COUNT
        high_beeps = self.DEFAULT_HIGH_BEEPS
        classification_modele = self.MOTIF_INJOIGNABLE
        behavior = ''
        
        for commande_res in commande_results:
            high_beeps = commande_res.high_beeps
            beep_count = commande_res.beep_count
            
            # Check for silence status with insufficient beeps
            if commande_res.classification.status.lower() == 'silence':
                if commande_res.beep_count < 5 and commande_res.high_beeps < 1:
                    is_compliant = self.STATUS_NON_CONFORM
                    commentaire += "Moins de 5 beeps trouvés. "
                break
            
            # Verify classification matches injoignable motif
            if commande_res.classification.status.lower() != self.MOTIF_INJOIGNABLE:
                is_compliant = self.STATUS_NON_CONFORM
                commentaire += f"Classification non conforme: {commande_res.classification.status}. "
                classification_modele = commande_res.classification.status.lower()
        
        return beep_count, high_beeps, classification_modele, behavior, is_compliant, commentaire
    
    def _process_non_injoignable_commande(
        self,
        commande_results: List[ComplianceInput],
        row_data: dict
    ) -> Tuple[int, int, str, str, str, str]:
        """
        Process compliance verification for non-'injoignable' motif cases.
        
        For non-injoignable cases, the method:
        - Uses values from the first result in the list
        - Extracts beep_count, high_beeps, and classification data
        - Compares the classification status with the current motif
        - Marks as non-compliant if classification doesn't match expected motif
        
        Args:
            commande_results: List of ComplianceInput results for this commande
            row_data: Dictionary containing the row data to update
            
        Returns:
            Tuple of (beep_count, high_beeps, classification_modele, behavior,
                     is_compliant, commentaire)
        """
        is_compliant = row_data.get('conformite_IAM', self.STATUS_CONFORM)
        commentaire = row_data.get('commentaire', '') or ''
        current_motif = row_data.get('motif_suspension', '')
        
        # Use first result for non-injoignable cases
        first_result = commande_results[0]
        beep_count = getattr(first_result, 'beep_count', 0)
        high_beeps = getattr(first_result, 'high_beeps', 0)
        classification_data = getattr(first_result, 'classification', None)
        
        # Extract classification attributes
        if classification_data:
            classification_modele = getattr(classification_data, 'status', '')
            behavior = getattr(classification_data, 'behavior', '')
        else:
            classification_modele = 'autre'
            behavior = ''
        
        # Verify classification matches expected motif
        if classification_modele != current_motif:
            is_compliant = self.STATUS_NON_CONFORM
            commentaire += f"Classification non conforme: {classification_modele.lower()}. "
        
        return beep_count, high_beeps, classification_modele, behavior, is_compliant, commentaire
    
    def _update_row_with_compliance_data(
        self,
        row_data: dict,
        beep_count: int,
        high_beeps: int,
        classification_modele: str,
        behavior: str,
        is_compliant: str,
        commentaire: str
    ) -> None:
        """
        Update row data dictionary with compliance verification results.
        
        Args:
            row_data: Dictionary to update (modified in place)
            beep_count: Number of beeps detected
            high_beeps: Number of high-pitched beeps detected
            classification_modele: Classification status from model
            behavior: Quality of communication behavior
            is_compliant: Compliance status (Conform/Non conforme)
            commentaire: Compliance comments
        """
        if classification_modele.lower() == "silence":
            classification_modele = "CLIENT INJOIGNABLE"
        row_data.update({
            'nb_tonnalite': beep_count,
            'high_beeps': high_beeps,
            'classification_modele': classification_modele,
            'qualite_communication': behavior,
            'conformite_IAM': is_compliant,
            'commentaire': commentaire.strip(),
            'processed': True
        })
    
    def _build_output_list(
        self,
        df_dict: List[dict],
        df_dict_by_commande: Dict[str, dict]
    ) -> List[dict]:
        """
        Build the final output list preserving original order from df_dict.
        
        Ensures all rows have proper types and default values for required fields.
        Rows that were updated have their beep_count and high_beeps converted to int.
        Rows without results get None values for beep fields.
        
        Args:
            df_dict: Original list of row dictionaries (preserves order)
            df_dict_by_commande: Dictionary of updated rows by numero_commande
            
        Returns:
            List of dictionaries with compliance data, preserving original order
        """
        updated_df_list = []
        for row in df_dict:
            numero_commande = row.get('numero_commande')
            if numero_commande and numero_commande in df_dict_by_commande:
                # Use updated row with proper type conversions
                updated_row = df_dict_by_commande[numero_commande]
                updated_row['nb_tonnalite'] = int(updated_row['nb_tonnalite']) if updated_row.get('nb_tonnalite') is not None else None
                updated_row['high_beeps'] = int(updated_row['high_beeps']) if updated_row.get('high_beeps') is not None else None
                updated_df_list.append(updated_row)
            else:
                # Row without results - set defaults
                row['nb_tonnalite'] = None
                row['high_beeps'] = None
                row['classification_modele'] = row.get('classification_modele', '')
                row['qualite_communication'] = row.get('qualite_communication', '')
                row['processed'] = False
                updated_df_list.append(row)
        return updated_df_list

    def verify_compliance_batch(
        self, 
        df_dict: List[dict], 
        results: List[ComplianceInput]
    ) -> List[dict]:
        """
        Verify compliance for a batch of commandes by merging classification results with original data.
        
        This method processes compliance verification results from audio classification and merges them
        with the original DataFrame rows. It handles two distinct cases based on the 'motif_suspension'
        field:
        
        1. **Injoignable Cases** (motif_suspension == 'injoignable'):
           - Iterates through all results for the commande
           - Checks for 'silence' status with insufficient beeps (< 5 beeps AND < 1 high beep)
           - Verifies all classifications match the 'injoignable' motif
           - Marks as non-compliant if any result has a different classification status
        
        2. **Non-Injoignable Cases** (all other motifs):
           - Uses the first result in the list for beep counts and classification
           - Extracts classification status and behavior from the first result
           - Compares classification status with the expected motif_suspension
           - Marks as non-compliant if classification doesn't match expected motif
        
        The method preserves the original order of rows in df_dict and ensures all rows have
        proper default values for required fields (beep_count, high_beeps, classification_modele,
        qualite_communication).
        
        Args:
            df_dict: List of dictionaries representing DataFrame rows. Each dictionary should contain:
                    - 'numero_commande': Order number (str, required for matching)
                    - 'motif_suspension': Suspension reason (str, determines processing logic)
                    - 'conformite_IAM': Initial compliance status (str, may be updated)
                    - 'commentaire': Initial comments (str, may be appended to)
                    Other fields may be present and will be preserved.
            
            results: List of ComplianceInput objects containing:
                    - numero_commande: Order number for matching with df_dict rows
                    - beep_count: Number of beeps detected in audio
                    - high_beeps: Number of high-pitched beeps detected
                    - classification: ClassificationResult with status and behavior
                    Multiple results may exist for the same numero_commande.
        
        Returns:
            List of dictionaries with updated compliance data. Each dictionary includes:
            - All original fields from df_dict
            - 'beep_count': Number of beeps (int or None)
            - 'high_beeps': Number of high beeps (int or None)
            - 'classification_modele': Classification status from model (str)
            - 'qualite_communication': Communication quality behavior (str)
            - 'conformite_IAM': Updated compliance status (str: 'Conform' or 'Non conforme')
            - 'commentaire': Updated comments with compliance issues (str)
            
            The list preserves the original order from df_dict. Rows without matching results
            in the results list will have None values for beep_count and high_beeps.
        
        Side Effects:
            - Logs processing statistics (number of unique commandes, completion status)
            - Logs warnings for commandes that are skipped (no results or not found in df_dict)
        
        
        """
        # Group results by numero_commande for efficient lookup
        results_by_commande = self._group_results_by_commande(results)
        logger.info(f'Processing {len(results_by_commande)} unique commandes from {len(results)} total results')
        
        # Create lookup dictionary for df_dict rows
        df_dict_by_commande = self._create_commande_lookup(df_dict)
        
        # Process each numero_commande that has results
        for numero_commande, commande_results in results_by_commande.items():
            # Skip if no results or numero_commande not found in lookup
            if not commande_results or numero_commande not in df_dict_by_commande:
                logger.warning(f'Skipping numero_commande {numero_commande}: no results or not found in df_dict')
                continue
            
            row_data = df_dict_by_commande[numero_commande]
            current_motif = row_data.get('motif_suspension')
            
            # Process based on motif type
            if current_motif.lower() == self.MOTIF_INJOIGNABLE:
                beep_count, high_beeps, classification_modele, behavior, is_compliant, commentaire = \
                    self._process_injoignable_commande(commande_results, row_data)
            else:
                beep_count, high_beeps, classification_modele, behavior, is_compliant, commentaire = \
                    self._process_non_injoignable_commande(commande_results, row_data)
            
            # Update row with computed compliance data
            self._update_row_with_compliance_data(
                row_data, beep_count, high_beeps, classification_modele,
                behavior, is_compliant, commentaire
            )
        
        # Build final output list preserving original order
        updated_df_list = self._build_output_list(df_dict, df_dict_by_commande)
        logger.info(f'Compliance verification completed for {len(updated_df_list)} commandes')
        return updated_df_list
