import pandas as pd
import re
from datetime import datetime
from typing import List, Optional, Tuple, Union
from sqlalchemy.orm import Session
from loguru import logger
from database.models import Call, Manifest, ManifestStatus
from database.database_manager import get_calls
import uuid


# Status filters for different manifest types
ACQUISITION_STATUSES = [
    'client injoignable',
    'client refuse installation',
    'client reporte rdv'
]

SAV_STATUSES = [
    'client injoignable',
    'client refuse installation',
    'client reporte rdv',
    'client absent',
    'absence routeur client',
    'local ferme'
]

RECYCLAGE_DATE_COLUMNS = [
    "premiere_date_recyclage",
    "deuxieme_date_recyclage",
    "troisieme_date_recyclage",
    "quatrieme_date_recyclage",
    "cinquieme_date_recyclage",
    "sixieme_date_recyclage"
]

SAV_RECYCLAGE_DATE_COLUMNS = [
    "date_commande",
    "date_reouverture_systeme",
    "date_recyclage"
]


def extract_contact_phone(comment: str) -> Optional[str]:
    """
    Extract phone number from commentaire_envoi_iam column.
    
    Rules:
    1. If "contact" appears, select the first mobile number after it
    2. If "contact" doesn't appear, select the first mobile number in the text
    3. Mobile numbers can be:
       - International format: 2126XXXXXXXX or 2127XXXXXXXX (12 digits)
       - Local format: 06XXXXXXXX or 07XXXXXXXX (10 digits)
    """
    if not comment or not isinstance(comment, str):
        return None
    
    # Match both international (2126/2127) and local (06/07) mobile formats
    mobile_pattern = r'(212[67]\d{8}|0[67]\d{8})'
    
    # First, check if "contact" appears in the text
    contact_match = re.search(r'contact', comment, re.IGNORECASE)
    
    if contact_match:
        # Search for the first mobile number after "contact"
        text_after_contact = comment[contact_match.end():]
        match = re.search(mobile_pattern, text_after_contact)
        if match:
            return normalize_phone_number(match.group(1))
    
    # Fallback: find the first mobile number in the entire text
    match = re.search(mobile_pattern, comment)
    return normalize_phone_number(match.group(1)) if match else None


def normalize_phone_number(phone: str) -> Optional[str]:
    """
    Normalize phone number to 9-digit format (without prefix).
    - '0XXXXXXXXX' → '212XXXXXXXXX' (add 212 prefix)
    """
    if not phone:
        return None

    if phone.startswith("0"):
        return '212'+phone[1:]
    return phone


class ManifestProcessor:
    def __init__(self, db_session: Session, config: dict):
        self.db = db_session
        self.config = config

    def process_manifest(self, csv_path: str, processing_date: Optional[datetime] = None) -> Tuple[pd.DataFrame, List[dict]]:
        """Parse a CSV manifest and return processed DataFrame."""
        manifest_id = str(uuid.uuid4())
        filename = csv_path.split('/')[-1]

        # Record manifest start
        manifest_record = Manifest(
            id=manifest_id,
            filename=filename,
            status=ManifestStatus.PROCESSING
        )
        self.db.add(manifest_record)
        self.db.commit()

        try:
            if 'crc_adsl' in filename.lower():
                df, calls_metadata = self._parse_acquisition(csv_path, processing_date, manifest_type='ADSL')
            elif 'crc_vula' in filename.lower():
                df, calls_metadata = self._parse_acquisition(csv_path, processing_date, manifest_type='VULA')
            elif 'sav' in filename.lower():
                df, calls_metadata = self._parse_sav(csv_path, processing_date)
            else:
                raise ValueError(f"Unsupported file type: {filename}")
    
            
            return df, calls_metadata

        except Exception as e:
            logger.error(f"Error processing manifest {csv_path}: {e}")
            manifest_record.status = ManifestStatus.FAILED
            self.db.commit()
            return pd.DataFrame(), []

    def _parse_acquisition(self, csv_path: str, date_suspension: Optional[datetime], manifest_type: str) -> Tuple[pd.DataFrame, List[Call]]:
        """Parse CRC ADSL or VULA CSV files (unified logic)."""
        config_mapping = self.config['csv_mappings']['acquisition'].get(manifest_type, {})
        
        # Try reading with multiple fallback strategies
        try:
            # Common case: UTF-8
            df = pd.read_csv(csv_path, encoding='utf-8', sep=None, engine='python')
        except (UnicodeDecodeError, pd.errors.ParserError):
            try:
                # Windows encodings with semicolon separator (common in French CSVs)
                df = pd.read_csv(csv_path, encoding='cp1252', sep=';')
            except (UnicodeDecodeError, pd.errors.ParserError):
                try:
                    # Try tab separation
                    df = pd.read_csv(csv_path, encoding='cp1252', sep='\t')
                except Exception:
                    # Final fallback: default separator, loose engine
                    df = pd.read_csv(csv_path, encoding='cp1252', sep=None, engine='python')
        
        # Rename columns
        df = df.rename(columns=config_mapping)
        logger.info(f"Initial rows: {len(df)}")
        # logger.info(f"Columns: {df.columns}")
        # Filter by status
        # Pre-convert date_suspension for filtering if it exists
        if "date_suspension" in df.columns:
            df["date_suspension"] = pd.to_datetime(df["date_suspension"], errors='coerce')

        df = self._filter_acquisition_status(df, date_suspension)
        logger.info(f"Rows after status filter: {len(df)}")
        
        # Deduplicate
        # Sort by client_number and then by date_suspension (newest/most recent last)
        df = df.sort_values(['client_number', 'date_suspension'], ascending=[False, False])
        # Drop duplicates based on client_number, keeping the most recent (last updated) date_suspension for each client
        df = df.drop_duplicates(subset='client_number', keep='first')
        
        # Convert date columns
        df = self._convert_date_columns(df, RECYCLAGE_DATE_COLUMNS + ["date_commande", "date_suspension"])
        
        # Adjust date_commande  based on recyclage dates
        df = self._adjust_date_commande_from_recyclage(df, manifest_type)
        
        # Normalize phone numbers
        df["client_number"] = df["client_number"].astype(str).apply(normalize_phone_number)
        logger.info(f"Rows with valid phone numbers: {len(df)}")
        
        calls_metadata = self._extract_calls_metadata(df)
        logger.info(f"Calls found: {len([c for c in calls_metadata if c])}")
        df = df[list(config_mapping.values())]
        df['Nbr_tentatives_appel'] = [len(c) for c in calls_metadata]
        return df, calls_metadata

    def _parse_sav(self, csv_path: str, date_suspension: Optional[datetime]) -> Tuple[pd.DataFrame, List[Call]]:
        """Parse SAV CSV file."""
        df = pd.read_csv(csv_path)
        df = df.rename(columns=self.config['csv_mappings']['SAV'])
        
        # Pre-convert date_suspension for filtering if it exists
        if "date_suspension" in df.columns:
            df["date_suspension"] = pd.to_datetime(df["date_suspension"], errors='coerce')
            
        # Filter by suspension status
        df = self._filter_sav_status(df, date_suspension)
        logger.info(f"Rows after SAV status filter: {len(df)}")
        
        # Extract contact phone from comments
        df["client_number"] = df["client_number"].apply(extract_contact_phone)
        df = df.dropna(subset=["client_number"])
        logger.info(f"Rows with valid extracted phone: {len(df)}")
        
        # Sort by client_number and then by date_suspension (newest/most recent last)
        df = df.sort_values(['client_number', 'date_suspension'], ascending=[False, False])
        # Drop duplicates based on client_number, keeping the most recent (last updated) date_suspension for each client
        df = df.drop_duplicates(subset='client_number', keep='first')
        
        # Convert date columns
        df = self._convert_date_columns(df, SAV_RECYCLAGE_DATE_COLUMNS + ["date_commande ", "date_suspension"])
        df = self._adjust_date_commande_from_recyclage(df, "SAV")

        calls_metadata = self._extract_calls_metadata(df)
        logger.info(f"Calls found (SAV): {len([c for c in calls_metadata if c])}")
        df = df[self.config['csv_mappings']['SAV'].values()]
        df['Nbr_tentatives_appel'] = [len(c) for c in calls_metadata]
        return df, calls_metadata

    def _filter_sav_status(self, df: pd.DataFrame, date_suspension: Union[datetime, str, None]) -> pd.DataFrame:
        """Filter SAV DataFrame by status conditions."""
        mask = (
            (df.status.str.lower().isin(SAV_STATUSES))
        )
        if date_suspension:
            # Ensure date_suspension arg is datetime
            if isinstance(date_suspension, str):
                date_suspension = pd.to_datetime(date_suspension)
            mask = mask & (df.date_suspension.dt.date == date_suspension.date())
        return df[mask].reset_index(drop=True)

    def _filter_acquisition_status(self, df: pd.DataFrame, date_suspension: Union[datetime, str, None]) -> pd.DataFrame:
        """Filter acquisition DataFrame by status conditions."""
        mask = (
            (df.status_commande.str.lower() == 'suspendue') &
            (df.status.str.lower().isin(ACQUISITION_STATUSES))
        )
        if date_suspension:
            # Ensure date_suspension arg is datetime
            if isinstance(date_suspension, str):
                date_suspension = pd.to_datetime(date_suspension)
            
            # Filter comparing dates
            mask = mask & (df.date_suspension.dt.date == date_suspension.date())
        return df[mask].reset_index(drop=True)

    def _convert_date_columns(self, df: pd.DataFrame, date_columns: List[str]) -> pd.DataFrame:
        """Convert date columns to datetime dtype."""
        
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        return df

    def _adjust_date_commande_from_recyclage(self, df: pd.DataFrame, manifest_type: str) -> pd.DataFrame:
        """
        Adjust date_commande  based on recyclage dates.
        Uses the most recent recyclage date that is between date_commande  and date_suspension.
        """
        def find_best_recyclage_date(row):
            for col in RECYCLAGE_DATE_COLUMNS if manifest_type == 'ADSL' else SAV_RECYCLAGE_DATE_COLUMNS:
                recyclage_date = row.get(col)
                if (
                    pd.notna(recyclage_date) and
                    row.date_commande  >= recyclage_date and
                    recyclage_date < row.date_suspension
                ):
                    return recyclage_date
            return row.date_commande 
        
        df["date_commande "] = df.apply(find_best_recyclage_date, axis=1)
        return df

    def _extract_calls_metadata(self, df: pd.DataFrame) -> List[List[dict]]:
        """Extract audio files from the manifest."""
        calls_metadata = []
        for _, row in df.iterrows():
            client_number = row.client_number
            date_commande = row.date_commande
            date_suspension = row.date_suspension
            

            # Logic for fetching calls
            strategy = 'all' if row.status.lower() == 'client injoignable' else 'first'
            
            calls = get_calls(self.db, client_number, date_commande, date_suspension, strategy=strategy)
            
            if calls is None:
                calls = []
            elif isinstance(calls, dict):
                calls = [calls]
            # Extend the calls_metadata list with the normalized list of calls (could be empty, one, or many)
            calls_metadata.append(calls)
            
        return calls_metadata
