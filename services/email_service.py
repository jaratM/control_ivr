import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import os
from loguru import logger
from typing import Dict, List

class EmailService:
    def __init__(self, config: Dict):
        """
        Initialize the EmailService with configuration.
        Expected config structure:
        email:
          smtp_host: str
          smtp_port: int
          smtp_user: str
          smtp_password: str
          sender_email: str
          recipient_emails: List[str]
          enabled: bool
        """
        self.email_config = config.get('email', {})
        self.enabled = self.email_config.get('enabled', False)
        self.smtp_host = self.email_config.get('smtp_host')
        self.smtp_port = self.email_config.get('smtp_port', 587)
        self.smtp_user = self.email_config.get('smtp_user')
        self.smtp_password = self.email_config.get('smtp_password')
        self.sender_email = self.email_config.get('sender_email')
        self.recipient_emails = self.email_config.get('recipient_emails', [])

    def send_result_email(self, file_path: str, manifest_type: str, category: str):
        """
        Send an email notification with the result CSV file attached.
        """
        if not self.enabled:
            logger.info("Email notifications are disabled.")
            return

        if not self.smtp_host or not self.recipient_emails:
            logger.warning("Email configuration missing smtp_host or recipient_emails. Skipping email.")
            return

        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}. Cannot send email.")
            return

        filename = os.path.basename(file_path)
        subject = f"Pipeline Results: {manifest_type} - {category} ({filename})"
        body = f"The pipeline has finished processing {manifest_type} {category}.\n\nPlease find the result CSV file attached."

        msg = MIMEMultipart()
        msg['From'] = self.sender_email
        msg['To'] = ", ".join(self.recipient_emails)
        msg['Subject'] = subject

        msg.attach(MIMEText(body, 'plain'))

        try:
            with open(file_path, "rb") as attachment:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment.read())
            
            encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition",
                f"attachment; filename= {filename}",
            )
            msg.attach(part)

            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                if self.smtp_user and self.smtp_password:
                    server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)
            
            logger.info(f"Email sent successfully with attachment: {filename}")
        except Exception as e:
            logger.error(f"Failed to send email: {str(e)}")
