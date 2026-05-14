from shared.config.settings import settings
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def test_email():
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = "PPU Assistant - Test Email"
        msg["From"] = settings.GMAIL_SENDER
        msg["To"] = settings.GMAIL_SENDER   # Send to yourself

        html = """
        <h2>✅ Email Test Successful!</h2>
        <p>If you received this email, Gmail SMTP is working correctly.</p>
        <p>PPU Assistant is ready to send verification codes.</p>
        """

        msg.attach(MIMEText(html, "html"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(settings.GMAIL_SENDER, settings.GMAIL_APP_PASSWORD)
            server.sendmail(settings.GMAIL_SENDER, settings.GMAIL_SENDER, msg.as_string())

        print("✅ Test email sent successfully! Check your inbox.")
        
    except Exception as e:
        print("❌ Failed to send email:")
        print(e)

if __name__ == "__main__":
    test_email()