from .types import ComplianceResult
import pandas as pd
from typing import List, Any, Dict
from loguru import logger
from database.models import Call
from datetime import timedelta, datetime

class ComplianceVerifier:
    def verify_compliance(self, df: pd.DataFrame, calls_metadata: List[List[Any]], manifest_type: str) -> pd.DataFrame:
        # Initialize lists to store results
        # We will append to these lists and assign them to the DF at the end to avoid SettingWithCopyWarning
        logger.info(f'dataframe shape: {df.shape}, calls_metadata shape: {len(calls_metadata)}')
        dispatched_calls_list = []
        branch_calls_list = []
        compliance_list = []
        comments_list = []

        # Iterate through the DataFrame
        for i, (_, row) in enumerate(df.iterrows()):
            # Default values for current row
            is_compliant = 'Conform'
            is_dispatched = 'Conform'
            is_branch_calls = 'Conform'
            comment = ""

            calls = calls_metadata[i]
            count = row.get('Nbr_tentatives_appel', 0)
            status = row.get('status', '')

            # Basic check: if no attempts recorded
            if count == 0:
                comment += "Aucun appel trouvé"
                is_compliant = 'Non conforme'
            
            else:
                # Helper to parse start_time (handles both datetime objects and ISO strings from serialization)
                def parse_start_time(call: Dict) -> datetime:
                    st = call.get('start_time')
                    if st is None:
                        return datetime.min
                    if isinstance(st, datetime):
                        return st
                    if isinstance(st, str):
                        return datetime.fromisoformat(st)
                    return datetime.min

                # Sort calls by start time to ensure correct order for gap checks
                sorted_calls = sorted(calls, key=parse_start_time)

                # Common Check: Verify all calls belong to the correct branch/manifest_type
                # This fixes the bug in the original code where 'i' was undefined in the else block
                for call in sorted_calls:
                    if manifest_type != call.get('branch'):
                        is_compliant = 'Non conforme'
                        is_branch_calls = 'Non conforme'
                        comment += f" Branche non conforme, "
                        break
                
                # Specific logic for 'client injoignable'
                if status.lower() == 'client injoignable':
                    # Check 1: Minimum number of attempts
                    if count < 3:
                        is_compliant = 'Non conforme'
                        comment += " Moins de 3 appels trouvés, "
                    
                    # Check 2: Time gap between consecutive calls
                    if len(sorted_calls) >= 2:
                        dispatched = True
                        for i in range(len(sorted_calls) - 1):
                            t1 = parse_start_time(sorted_calls[i])
                            t2 = parse_start_time(sorted_calls[i+1])
                            diff = t2 - t1
                            
                            # 2 hours = 7200 seconds
                            if diff.total_seconds() < 7200:
                                is_dispatched = 'Non conforme'
                                is_compliant = 'Non conforme'
                                hours_diff = diff.total_seconds() / 3600
                                if dispatched:
                                    comment += f" Le temps entre les appels {i+1} et {i+2} est de {hours_diff:.2f} heures, moins de 2 heures"
                                    dispatched = False

            # Store results for this row
            dispatched_calls_list.append(is_dispatched)
            branch_calls_list.append(is_branch_calls)
            compliance_list.append(is_compliant)
            comments_list.append(comment)

        # Bulk assign columns to DataFrame
        df['appels_dispatches'] = dispatched_calls_list
        df['appels_branch'] = branch_calls_list
        df['compliance'] = compliance_list
        df['commentaires'] = comments_list

        return df
