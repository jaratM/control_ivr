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
    STATUS_CONFORM = 'Conforme'
    STATUS_NON_CONFORM = 'Non conforme'
    CLIENT_UNREACHABLE = 'client injoignable'
    MANIFEST_ACQUISITION = 'ACQUISITION'
    MANIFEST_SAV = 'SAV'
    MIN_ATTEMPTS = 3
    MIN_GAP_SECONDS = 7200 # 2 hours

    MOTIF_INJOIGNABLE = 'client injoignable'


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

    def get_time(self, res: ComplianceInput) -> datetime:
        return self._parse_date_value(res.metadata.start_time)
    
    def get_effective_status(self, res: ComplianceInput) -> Tuple[str, bool]:
            status = res.classification.status.strip().lower()
            nb_beeps_ok = True

            if status == "silence":
                if res.beep_count < 5 and res.beep_count > 0 and res.high_beeps < 1:
                    nb_beeps_ok = False
                status = self.CLIENT_UNREACHABLE

            return status, nb_beeps_ok

    def _process_injoignable_commande(
        self,
        commande_results: List[ComplianceInput],
        row_data: dict
    ) -> Tuple[int, int, str, str, str, str, bool, bool, datetime]:
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
                     is_compliant, commentaire, has_consecutive_calls)
        """
        is_compliant = row_data.get('conformite_IAM', self.STATUS_CONFORM)
        commentaire = row_data.get('commentaire', '') or ''
        has_consecutive_calls = False
        conformite_nb_beeps = True
        # Helper to get start time from result
        

        # Sort results by start time
        commande_results.sort(key=self.get_time, reverse=True)
        
        # Helper to get effective status (handling silence logic)
        

        last_result = commande_results[0]
        beep_count = last_result.beep_count
        high_beeps = last_result.high_beeps
        behavior = last_result.classification.behavior
        classification_modele, _ = self.get_effective_status(last_result)

        # 1. Check if last call is unreachable
        if classification_modele != self.CLIENT_UNREACHABLE:
            is_compliant = self.STATUS_NON_CONFORM
            commentaire += f" Dernier appel non conforme (Statut: {classification_modele}). "
            return beep_count, high_beeps, classification_modele, behavior, is_compliant, commentaire, False, False, self.get_time(last_result)   
        
        if len(commande_results) >= 3:
            for i in range(len(commande_results) - 2):
                t1 = self.get_time(commande_results[i])
                t2 = self.get_time(commande_results[i+1])
                t3 = self.get_time(commande_results[i+2])
                c_m_1, nb_beeps_ok_1 = self.get_effective_status(commande_results[i])
                c_m_2, nb_beeps_ok_2 = self.get_effective_status(commande_results[i+1])
                c_m_3, nb_beeps_ok_3 = self.get_effective_status(commande_results[i+2])
                diff1 = (t1-t2).total_seconds()
                diff2 = (t2-t3).total_seconds()
                
                if diff1 >= self.MIN_GAP_SECONDS and diff2 >= self.MIN_GAP_SECONDS and c_m_1 == self.CLIENT_UNREACHABLE and c_m_2 == self.CLIENT_UNREACHABLE and c_m_3 == self.CLIENT_UNREACHABLE:
                    has_consecutive_calls = True
                    if not nb_beeps_ok_1 or not nb_beeps_ok_2 or not nb_beeps_ok_3:
                        beep_count = min(commande_results[i].beep_count, commande_results[i+1].beep_count, commande_results[i+2].beep_count)
                        is_compliant = self.STATUS_NON_CONFORM
                        commentaire += ", Moins de 5 beeps"
                        conformite_nb_beeps = False
                    break
        else:
            call_time = self.get_time(last_result)
            date_commande = row_data.get('date_commande', '')
            date_commande = self._parse_date_value(date_commande)
            # Last call must be after 5pm (17:00) to be compliant
            if date_commande.hour > 17 and call_time.hour > 17 and date_commande.date() == call_time.date():
                commentaire += " Dernier appel effectué apres 17h"
            else:
                is_compliant = self.STATUS_NON_CONFORM
                commentaire += f" Moins de {self.MIN_ATTEMPTS} appels trouvés"
             
        if not has_consecutive_calls:
            is_compliant = self.STATUS_NON_CONFORM
            commentaire += ", Pas de sequence de 3 appels avec 2h d'ecart. "
            

        return beep_count, high_beeps, classification_modele, behavior, is_compliant, commentaire, has_consecutive_calls, conformite_nb_beeps, self.get_time(last_result)
    
    def _process_non_injoignable_commande(
        self,
        commande_results: List[ComplianceInput],
        row_data: dict
    ) -> Tuple[int, int, str, str, str, str, datetime]:
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
                     is_compliant, commentaire, date_appel)
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
        classification_modele, _ = self.get_effective_status(first_result)

        # Verify classification matches expected motif
        
        if classification_modele.lower() != current_motif.lower():
            is_compliant = self.STATUS_NON_CONFORM
            commentaire += f" Declaration non conforme: {current_motif.lower()}. "
        
        return beep_count, high_beeps, classification_modele, behavior, is_compliant, commentaire, self.get_time(first_result)
    
    def _update_row_with_compliance_data(
        self,
        row_data: dict,
        beep_count: int,
        high_beeps: int,
        classification_modele: str,
        behavior: str,
        is_compliant: str,
        commentaire: str,
        has_consecutive_calls: bool,
        conformite_nb_beeps: bool,
        date_appel: datetime
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
            has_consecutive_calls: Whether there are 3 consecutive calls with 2h interval
            conformite_nb_beeps: Whether the number of beeps is compliant
            date_appel: Date of the last call
        """
        if classification_modele.lower() == "silence":
            classification_modele = "CLIENT INJOIGNABLE"
        row_data.update({
            'date_appel_technicien': date_appel,
            'nb_tonnalite': beep_count,
            'high_beeps': high_beeps,
            'classification_modele': classification_modele,
            'qualite_communication': behavior,
            'conformite_IAM': is_compliant,
            'commentaire': commentaire.strip(),
            'processed': True,
            'conformite_intervalle': self.STATUS_CONFORM if has_consecutive_calls else self.STATUS_NON_CONFORM,
            'conformite_nb_beeps': self.STATUS_CONFORM if conformite_nb_beeps else self.STATUS_NON_CONFORM
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
                row['processed'] = True
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
                # logger.warning(f'Skipping numero_commande {numero_commande}: no results or not found in df_dict')
                continue
            
            row_data = df_dict_by_commande[numero_commande]
            current_motif = row_data.get('motif_suspension')
            has_consecutive_calls = False
            conformite_nb_beeps = True
            beep_count = None   
            high_beeps = None
            classification_modele = None
            behavior = None
            is_compliant = None
            commentaire = None
            date_appel = None
            if len(commande_results) == 0:
                self._update_row_with_compliance_data(
                    row_data, None, None, None, None, None, 'Aucun résultat trouvé', False, True, None
                )
                continue
            # Process based on motif type
            if current_motif and current_motif.lower() == self.MOTIF_INJOIGNABLE:
                beep_count, high_beeps, classification_modele, behavior, is_compliant, commentaire, has_consecutive_calls, conformite_nb_beeps, date_appel = \
                    self._process_injoignable_commande(commande_results, row_data)
            else:
                beep_count, high_beeps, classification_modele, behavior, is_compliant, commentaire, date_appel = \
                    self._process_non_injoignable_commande(commande_results, row_data)
            
            # Update row with computed compliance data
            self._update_row_with_compliance_data(
                row_data, beep_count, high_beeps, classification_modele,
                behavior, is_compliant, commentaire, has_consecutive_calls, conformite_nb_beeps,
                date_appel
            )
            
        # Build final output list preserving original order
        updated_df_list = self._build_output_list(df_dict, df_dict_by_commande)
        logger.info(f'Compliance verification completed for {len(updated_df_list)} commandes')
        return updated_df_list


