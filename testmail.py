import logging
import smtplib
import ssl
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText



def configure_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename='email_sender.log',
        filemode='a'
    )

    # Define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    # Set a format that is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)

    # Add the handler to the root logger
    logging.getLogger('').addHandler(console)


# Configure the logging
configure_logging()

# Create a logger for this module
logger = logging.getLogger(__name__)


def send_email_with_attachment(sender_email, to_emails, cc_emails, password, subject, body, filename):
    logger.info('Preparing to send email...')
    logger.info('Sender: %s', sender_email)
    logger.info('To: %s', ', '.join(to_emails))
    logger.info('Cc: %s', ', '.join(cc_emails))
    logger.info('Subject: %s', subject)
    logger.info('Attachment: %s', filename)

    # Create a multipart message
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = ", ".join(to_emails)
    message["Cc"] = ", ".join(cc_emails)
    message["Subject"] = subject

    # Add body to email
    message.attach(MIMEText(body, "plain"))
    logger.debug('Email body attached')

    # Open the file in binary mode
    try:
        with open(filename, "rb") as attachment:
            # Add file as application/octet-stream
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())
        logger.debug('File %s read successfully', filename)
    except IOError as e:
        logger.error('Failed to read attachment file: %s', e)
        raise

    # Encode file in ASCII characters to send by email
    encoders.encode_base64(part)
    logger.debug('File encoded successfully')

    # Add header as key/value pair to attachment part
    part.add_header(
        "Content-Disposition",
        f"attachment; filename= {filename}",
    )

    # Add attachment to message
    message.attach(part)
    logger.debug('Attachment added to message')

    # Convert message to string
    text = message.as_string()

    # Combine all recipients
    all_recipients = to_emails + cc_emails

    # Log in to server using secure context and send email
    context = ssl.create_default_context()
    try:
        with smtplib.SMTP("mail.dccmail.in", 587) as server:
            logger.info('Connecting to SMTP server...')
            server.ehlo()  # Can be omitted
            server.starttls(context=context)
            server.ehlo()  # Can be omitted
            server.login(sender_email, password)
            logger.info('Logged in successfully')
            server.sendmail(sender_email, all_recipients, text)
            logger.info('Email sent successfully!')
    except smtplib.SMTPException as e:
        logger.error('An error occurred while sending the email: %s', e)
        raise

    logger.info('Email sending process completed')


if __name__ == "__main__":
    # Usage example
    sender_email = "systemalert@mail.dccmail.in"
    to_emails = ["user.harpreetsingh@gmail.com", "jatinder.singh@mail.dccmail.in"]
    cc_emails = ["monish.chopra@mail.dccmail.in", "anurag@mail.dccmail.in"]
    password = "ZUqp2215@%"
    subject = "Email with Attachment"
    body = "Please find the attached file."
    filename = "anpr_log.txt"  # Make sure this file exists in the same directory as the script

    try:
        send_email_with_attachment(sender_email, to_emails, cc_emails, password, subject, body, filename)
    except Exception as e:
        logger.exception("An error occurred while sending the email:")