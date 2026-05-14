# gateway/api/auth/email_sender.py
import smtplib
import random
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from shared.config.settings import settings

def generate_code():
    return str(random.randint(100000, 999999))

def send_verification_email(to_email: str, code: str):
    subject = "PPU Assistant - Verification Code"

    html = f"""
    <h2>Your Verification Code</h2>
    <h1 style="font-size:48px; letter-spacing:10px;">{code}</h1>
    <p>This code expires in {settings.VERIFICATION_CODE_EXPIRE_MINUTES} minutes.</p>
    """

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = settings.GMAIL_SENDER
    msg["To"] = to_email
    msg.attach(MIMEText(html, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(settings.GMAIL_SENDER, settings.GMAIL_APP_PASSWORD)
        server.sendmail(settings.GMAIL_SENDER, to_email, msg.as_string())