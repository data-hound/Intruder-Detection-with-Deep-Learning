import smtplib
import getpass
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart

GMAIL_USER = raw_input('your_name@gmail.com:')
GMAIL_PASS = raw_input('enter your password:')#getpass.getpass('your_password')
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
def send_email(recipient, subject, text):
    smtpserver = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
    smtpserver.ehlo()
    smtpserver.starttls()
    smtpserver.ehlo
    smtpserver.login(GMAIL_USER, GMAIL_PASS)
    header = 'To:' + recipient + '\n' + 'From: ' + GMAIL_USER
    header = header + '\n' + 'Subject:' + subject + '\n'
    msg = header + '\n' + text + ' \n\nsa'
    smtpserver.sendmail(GMAIL_USER, recipient, msg)
    smtpserver.close()
send_email('sabathanshuman@gmail.com', 'sub', 'this is text')

