from fileinput import filename
from unicodedata import category
import pandas as pd
import re
from datetime import datetime
from typing import List, NamedTuple, Optional, Tuple, Union
from sqlalchemy.orm import Session
from loguru import logger
from database.models import Call, Manifest, ManifestStatus, ManifestCall
from database.database_manager import get_calls, get_manifest
import uuid
from database.database_manager import update_manifest_status


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



def extract_contact_name(comment: str) -> Optional[str]:
    """
    Extract client name (nom_prenom) from commentaire_envoi_iam column.
    
    Handles various formats:
    - Nom et prénom :HADIR MUSTAPHA Numéro...
    - *Nom et prénom client : ALI BOKADDA *Numéro...
    - Nom client :HANANE JABER Numéro...
    - Nom & prénom : ABDERRAZAK MOUKRIM Numéro...
    - ABDELHAK NAJI Numéro de contact : 212... (name at start without label)
    - numéro de contact : 212704356193 SAID AMAST (name at end after phone)
    - N° de contact :212630192253 YASMINA KORAICHI (name at end after phone)
    
    Returns:
        The extracted name or None if not found.
    """
    if not comment or not isinstance(comment, str):
        return None
    
    comment = comment.strip()
    
    # Strategy 1: Try to find name after a label (Nom...:)
    name = _extract_name_after_label(comment)
    if name:
        return name
    
    # Strategy 2: Try to find name before phone number pattern (name at start)
    name = _extract_name_before_phone(comment)
    if name:
        return name
    
    # Strategy 3: Try to find name after phone number (name at end)
    name = _extract_name_after_phone(comment)
    if name:
        return name
    
    return None


def _extract_name_after_label(comment: str) -> Optional[str]:
    """Extract name that appears after a label like 'Nom et prénom :'"""
    # Pattern to match various name label formats
    name_label_pattern = r'\*?\s*Nom\s*(?:et\s+prénom|&\s*prénom|client)?(?:\s+client)?\s*:\s*'
    
    # Patterns that indicate the end of the name field
    end_patterns = [
        r'\*?\s*Numéro',      # "Numéro de la ligne", "*Numéro de contact", etc.
        r'\*?\s*N°',          # "N° de contact"
        r'\*?\s*Tel',         # "Tel:", "Téléphone"
        r'\*?\s*numéro',      # lowercase "numéro"
    ]
    end_pattern = '|'.join(end_patterns)
    
    label_match = re.search(name_label_pattern, comment, re.IGNORECASE)
    if not label_match:
        return None
    
    text_after_label = comment[label_match.end():]
    end_match = re.search(end_pattern, text_after_label, re.IGNORECASE)
    
    if end_match:
        name = text_after_label[:end_match.start()].strip()
    else:
        name = text_after_label.strip()
        # If there's a lot of text, cut at first phone number
        if len(name) > 50:
            num_match = re.search(r'\d{3,}', name)
            if num_match:
                name = name[:num_match.start()].strip()
    
    return _clean_and_validate_name(name)


def _extract_name_before_phone(comment: str) -> Optional[str]:
    """Extract name that appears at the start, before any phone-related text."""
    # Pattern: text at start followed by phone-related keywords
    # e.g., "ABDELHAK NAJI Numéro de contact : 212..."
    phone_keyword_pattern = r'(?:\*?\s*(?:Numéro|N°|numéro)\s*(?:de\s+)?(?:contact|ligne|la\s+ligne)?e?\s*:)'
    
    match = re.search(phone_keyword_pattern, comment, re.IGNORECASE)
    if not match:
        return None
    
    # Get text before the phone keyword
    text_before = comment[:match.start()].strip()
    
    # Remove any leading asterisks or special chars
    text_before = re.sub(r'^[\*\s]+', '', text_before)
    
    # Check if this looks like a name (alphabetic characters, no phone numbers)
    if text_before and not re.search(r'\d{6,}', text_before):
        return _clean_and_validate_name(text_before)
    
    return None


def _extract_name_after_phone(comment: str) -> Optional[str]:
    """Extract name that appears after a phone number at the end."""
    # Pattern: phone number followed by name at end
    # e.g., "N° de contact :212630192253 YASMINA KORAICHI"
    # e.g., "numéro de contact : 212704356193 SAID AMAST"
    
    # Find phone numbers (10-12 digits)
    phone_pattern = r'(?:212[67]\d{8}|0[67]\d{8})'
    
    matches = list(re.finditer(phone_pattern, comment))
    if not matches:
        return None
    
    # Get text after the last phone number
    last_match = matches[-1]
    text_after = comment[last_match.end():].strip()
    
    # Clean up: remove leading punctuation, whitespace
    text_after = re.sub(r'^[\s\*\:\-]+', '', text_after)
    
    # Check if remaining text looks like a name
    if text_after and not re.search(r'\d{6,}', text_after):
        return _clean_and_validate_name(text_after)
    
    return None


def _clean_and_validate_name(name: str) -> Optional[str]:
    """Clean up and validate extracted name."""
    if not name:
        return None
    
    # Remove trailing punctuation or extra whitespace
    name = re.sub(r'[\*\:\-\,\.]+$', '', name).strip()
    
    # Remove leading punctuation
    name = re.sub(r'^[\*\:\-\,\.]+', '', name).strip()
    
    # Validate: name should contain at least some alphabetic characters
    # and should be reasonable length (2-60 chars)
    if name and 2 <= len(name) <= 60 and re.search(r'[A-Za-zÀ-ÿ]{2,}', name):
        return name
    
    return None





class ManifestProcessor:
    def __init__(self, db_session: Session, config: dict):
        self.db = db_session
        self.config = config

    def process_manifest(self, csv_path: str, processing_date: Optional[datetime] = None, manifest_record: Manifest = None) -> Tuple[List[dict], List[dict], Manifest]:
        """Parse a CSV manifest and return processed DataFrame and calls metadata."""
        # Record manifest start
        filename = csv_path.split('/')[-1]

        manifest_type = ''
        category = ''
        try:
            # Acquisition adsl csv
            if 'crc_adsl' in filename.lower():
                df, calls_metadata = self._parse_acquisition(csv_path, processing_date, category='ADSL')
                manifest_type = 'ACQUISITION'
                category = 'ADSL'
                df['categorie'] = category
                df['line_id'] = None
                df['nom_prenom'] = None
            # Acquisition vuLA csv
            elif 'crc_vula' in filename.lower():
                df, calls_metadata = self._parse_acquisition(csv_path, processing_date, category='VULA')
                manifest_type = 'ACQUISITION'
                category = 'VULA'
                df['categorie'] = category
                df['numero_ordre'] = None
                df['nom_prenom'] = None
            # SAV csv
            elif 'sav' in filename.lower():
                df, calls_metadata = self._parse_sav(csv_path, processing_date)
                manifest_type = 'SAV'
                # df['numero_ordre'] = None
                df['line_id'] = None
                df['categorie'] = df['categorie'].apply(lambda x: 'VULA' if x == 'ftth' else 'ADSL')
            else:
                raise ValueError(f"Unsupported file type: {filename}")
            
            # Convert DataFrame to list of dicts, default to empty list if df is None
            if df is not None:
                df['manifest_id'] = manifest_record.id
                # logger.info(f"Processing {filename}: Columns: {df.columns}")
                df_dict = df.to_dict(orient='records')
            else:
                df_dict = []
                
            return df_dict, calls_metadata, manifest_type, category

        except Exception as e:
            logger.error(f"Error processing manifest {csv_path}: {e}")
            update_manifest_status(self.db, manifest_record.id, ManifestStatus.FAILED)
            return [], [], None, None

    def _parse_acquisition(self, csv_path: str, date_suspension: Optional[datetime], category: str) -> Tuple[pd.DataFrame, List[Call]]:
        """
        Parse ACQUISITION ADSL or VULA CSV files (unified logic).
        - Read the CSV file
        - Rename columns
        - Filter by status
        - Convert date columns
        - Adjust date_commande  based on recyclage dates
        - Extract calls metadata
        - Only keep columns that exist in the dataframe
        - Return the dataframe and calls metadata
        """
        config_mapping = self.config['csv_mappings']['acquisition'].get(category, {})
        
        # Try reading with multiple fallback strategies
        df = self._read_df(csv_path)
        # Rename columns
        df = df.rename(columns=config_mapping)
        logger.info(f"Processing Acquisition: Initial rows: {len(df)}\n")
        # logger.info(f"Columns: {df.columns}")
        # Filter by status
        # Pre-convert date_suspension for filtering if it exists
        if "date_suspension" in df.columns:
            df["date_suspension"] = pd.to_datetime(df["date_suspension"], errors='coerce')

        df = self._filter_acquisition_status(df, date_suspension)
        logger.info(f"Processing Acquisition: Rows after status filter: {len(df)}\n")
        
        # Deduplicate
        # Sort by client_number and then by date_suspension (newest/most recent last)
        df = df.sort_values(['client_number', 'date_suspension'], ascending=[False, False])
        # Drop duplicates based on client_number, keeping the most recent (last updated) date_suspension for each client
        df = df.drop_duplicates(subset='client_number', keep='first')
        
        # Convert date columns
        df = self._convert_date_columns(df, RECYCLAGE_DATE_COLUMNS + ["date_commande", "date_suspension"])
        
        # Adjust date_commande  based on recyclage dates
        df = self._adjust_date_commande_from_recyclage(df, category)
        
        # Normalize phone numbers
        df["client_number"] = df["client_number"].astype(str).apply(normalize_phone_number)
        logger.info(f"Processing Acquisition: Rows with valid phone numbers: {len(df)}\n")
        
        calls_metadata = self._extract_calls_metadata(df)
        logger.info(f"Processing Acquisition: Calls found: {len([c for c in calls_metadata if c])}\n")
        
        # Only keep columns that exist in the dataframe
        available_columns = [col for col in config_mapping.values() if col in df.columns and col != "status_commande"]
        
        
        df = df[available_columns]
        
        df['nbr_tentatives_appel'] = [len(c) for c in calls_metadata]
        return df, calls_metadata

    def _parse_sav(self, csv_path: str, date_suspension: Optional[datetime]) -> Tuple[pd.DataFrame, List[Call]]:
        """
        Parse SAV CSV file (unified logic ).
        - Filter by status
        - Extract contact phone from comments
        - Sort by client_number and then by date_suspension (newest/most recent last)
        - Drop duplicates based on client_number, keeping the most recent (last updated) date_suspension for each client
        - Convert date columns
        - Adjust date_commande  based on recyclage dates
        - Extract calls metadata
        - Only keep columns that exist in the dataframe
        - Return the dataframe and calls metadata
        """

        df = self._read_df(csv_path)
        df = df.rename(columns=self.config['csv_mappings']['SAV'])
        
        # Pre-convert date_suspension for filtering if it exists
        if "date_suspension" in df.columns:
            df["date_suspension"] = pd.to_datetime(df["date_suspension"], errors='coerce')
            
        # Filter by suspension status
        df = self._filter_sav_status(df, date_suspension)
        logger.info(f"Processing SAV: Rows after SAV status filter: {len(df)}\n")

        # Extract name and phone from the contact/comment field
        df["client_number"] = df["contact"].apply(extract_contact_phone)
        df["nom_prenom"] = df["contact"].apply(extract_contact_name)
        df = df.drop(columns=["contact"], errors='ignore')
        
        df = df.dropna(subset=["client_number"])
        logger.info(f"Processing SAV: Rows with valid extracted phone: {len(df)}\n")
        
        # Sort by client_number and then by date_suspension (newest/most recent last)
        df = df.sort_values(['client_number', 'date_suspension'], ascending=[False, False])
        # Drop duplicates based on client_number, keeping the most recent (last updated) date_suspension for each client
        df = df.drop_duplicates(subset='client_number', keep='first')
        
        # Convert date columns
        df = self._convert_date_columns(df, SAV_RECYCLAGE_DATE_COLUMNS + ["date_commande", "date_suspension"])
        df = self._adjust_date_commande_from_recyclage(df, "SAV")

        calls_metadata = self._extract_calls_metadata(df)
        logger.info(f"Processing SAV: Calls found (SAV): {len([c for c in calls_metadata if c])}\n")
        
        # Only keep columns that exist in the dataframe
        available_columns = [col for col in self.config['csv_mappings']['SAV'].values() if col in df.columns and col != "date_recyclage"]
        df = df[available_columns]
        
        df['nbr_tentatives_appel'] = [len(c) for c in calls_metadata]
        return df, calls_metadata

    def _read_df(self, file_path: str) -> pd.DataFrame:
        """Read file and return DataFrame.
        - Read the CSV file
        - Read the file as an Excel file
        - Return the dataframe
        """
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:   
            try:
            # Common case: UTF-8
                df = pd.read_csv(file_path, encoding='utf-8', sep=None, engine='python')
            except (UnicodeDecodeError, pd.errors.ParserError):
                try:
                    # Windows encodings with semicolon separator (common in French CSVs)
                    df = pd.read_csv(file_path, encoding='cp1252', sep=';')
                except (UnicodeDecodeError, pd.errors.ParserError):
                    try:
                        # Try tab separation
                        df = pd.read_csv(file_path, encoding='cp1252', sep='\t')
                    except Exception:
                        # Final fallback: default separator, loose engine
                        df = pd.read_csv(file_path, encoding='cp1252', sep=None, engine='python')
        return df

    def _filter_sav_status(self, df: pd.DataFrame, date_suspension: Union[datetime, str, None]) -> pd.DataFrame:
        """
        Filter SAV DataFrame by status conditions.
        - Filter by status
        - Filter by date_suspension
        - Return the dataframe
        """
        mask = (
            (df.motif_suspension.str.lower().isin(SAV_STATUSES))
        )
        if date_suspension:
            # Ensure date_suspension arg is datetime
            if isinstance(date_suspension, str):
                date_suspension = pd.to_datetime(date_suspension)
            mask = mask & (df.date_suspension.dt.date == date_suspension.date())
        return df[mask].reset_index(drop=True)

    def _filter_acquisition_status(self, df: pd.DataFrame, date_suspension: Union[datetime, str, None]) -> pd.DataFrame:
        """
        Filter acquisition DataFrame by status conditions.
        - Filter by status
        - Filter by date_suspension
        - Return the dataframe
        """
        mask = (
            (df.status_commande.str.lower() == 'suspendue') &
            (df.motif_suspension.str.lower().isin(ACQUISITION_STATUSES))
        )
        if date_suspension:
            # Ensure date_suspension arg is datetime
            if isinstance(date_suspension, str):
                date_suspension = pd.to_datetime(date_suspension)
            
            # Filter comparing dates
            mask = mask & (df.date_suspension.dt.date == date_suspension.date())
        return df[mask].reset_index(drop=True)

    def _convert_date_columns(self, df: pd.DataFrame, date_columns: List[str]) -> pd.DataFrame:
        """
        Convert date columns to datetime dtype.
        Return the dataframe
        """
        
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
        
        df["date_commande"] = df.apply(find_best_recyclage_date, axis=1)
        return df

    def _extract_calls_metadata(self, df: pd.DataFrame) -> List[List[dict]]:
        """
        Extract calls metadata from the manifest.
        - Extract the client number
        - Extract the date_commande
        - Extract the date_suspension
        - Get the calls based on the client number and date range
        - Normalize the calls metadata
        - Return the calls metadata
        """
        calls_metadata = []
        for _, row in df.iterrows():
            client_number = row.client_number
            date_commande = row.date_commande
            date_suspension = row.date_suspension
            numero_commande = row.numero_commande

            # Logic for fetching calls
            strategy = 'all' if row.motif_suspension.lower() == 'client injoignable' else 'last'
            
            calls = get_calls(self.db, client_number, date_commande, date_suspension, numero_commande, strategy=strategy)
            
            if calls is None:
                calls = []
            elif isinstance(calls, dict):
                calls = [calls]
            # Extend the calls_metadata list with the normalized list of calls (could be empty, one, or many)
            calls_metadata.append(calls)
            
        return calls_metadata
