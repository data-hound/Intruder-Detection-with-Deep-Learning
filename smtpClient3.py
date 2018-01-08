import smtplib
import getpass
import os
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart

GMAIL_USER = raw_input('your_name@gmail.com:')
GMAIL_PASS = raw_input('enter your password:')#getpass.getpass('your_password')
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587

def SendMail(ImgFileName):
    img_data = open(ImgFileName, 'rb').read()
    msg = MIMEMultipart()
    msg['Subject'] = 'subject'
    msg['From'] = GMAIL_USER
    msg['To'] = 'sabathanshuman@gmail.com'

    text = MIMEText("test")
    msg.attach(text)
    image = MIMEImage(img_data, name=os.path.basename(ImgFileName))
    msg.attach(image)

    s = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
    s.ehlo()
    s.starttls()
    s.ehlo()
    s.login(GMAIL_USER, GMAIL_PASS)
    s.sendmail(GMAIL_USER,'sabathanshuman@gmail.com', msg.as_string())
    s.quit()

