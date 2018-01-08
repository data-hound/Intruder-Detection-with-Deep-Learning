import smtplib
import email.utils
from email.mime.text import MIMEText

# Create the message
msg = MIMEText('This is the body of the message.')
msg['To'] = email.utils.formataddr(('Recipient', 'sabathanshuman@gmail.com'))
msg['From'] = email.utils.formataddr(('Author', 'deepDreamer1729@gmail.com'))
msg['Subject'] = 'Simple test message'

server = smtplib.SMTP('locaalhost', 25)
server.set_debuglevel(True) # show communication with the server
try:
    server.sendmail('sabathanshuman@gmail.com', ['deepDreamer1729@gmail.com'], msg.as_string())
finally:
    server.quit()
