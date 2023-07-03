import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def send_verification_email(email: str):
    # Конфигурация SMTP сервера
    smtp_host = "your_smtp_host"
    smtp_port = 587  # Порт SMTP сервера
    smtp_username = "your_username"
    smtp_password = "your_password"

    # Создание объекта сообщения
    msg = MIMEMultipart()
    msg["From"] = "your_email@example.com"
    msg["To"] = email
    msg["Subject"] = "Подтверждение регистрации"

    # Создание текста сообщения
    body = """
    Здравствуйте,

    Пожалуйста, подтвердите свою регистрацию, перейдя по ссылке ниже:

    http://your_website.com/verify-email?token=your_verification_token

    Спасибо.
    """
    msg.attach(MIMEText(body, "plain"))

    # Подключение к SMTP серверу и отправка сообщения
    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls()
        server.login(smtp_username, smtp_password)
        server.send_message(msg)
