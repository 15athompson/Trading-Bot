import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import config

class NotificationSystem:
    def __init__(self):
        self.email = config.NOTIFICATION_EMAIL
        self.password = config.NOTIFICATION_EMAIL_PASSWORD
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587

    def send_email(self, subject, body, to_email):
        msg = MIMEMultipart()
        msg['From'] = self.email
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        try:
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.email, self.password)
            text = msg.as_string()
            server.sendmail(self.email, to_email, text)
            server.quit()
            print(f"Email sent successfully to {to_email}")
        except Exception as e:
            print(f"Error sending email: {str(e)}")

    def send_alert(self, alert_type, message):
        subject = f"Crypto Trading Bot Alert: {alert_type}"
        self.send_email(subject, message, config.ALERT_EMAIL)

notifier = NotificationSystem()

# Usage examples:
# notifier.send_alert("High Volatility", "BTC/USDT volatility has exceeded 5% in the last hour")
# notifier.send_alert("Performance Milestone", "Bot has achieved 10% return on investment")