import os
from pathlib import Path
import ssl
import smtplib

# Load .env if present
base = Path(__file__).resolve().parent
env = base / '.env'
if env.exists():
    with env.open('r', encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line or line.startswith('#') or '=' not in line: continue
            k,v=line.split('=',1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

EMAIL_SMTP_SERVER = os.environ.get('EMAIL_SMTP_SERVER','smtp.gmail.com')
EMAIL_SMTP_PORT = int(os.environ.get('EMAIL_SMTP_PORT','587'))
EMAIL_SENDER = os.environ.get('EMAIL_SENDER')
EMAIL_PASSWORD = os.environ.get('EMAIL_PASSWORD')

print('SMTP server:', EMAIL_SMTP_SERVER)
print('SMTP port:', EMAIL_SMTP_PORT)
print('Sender:', EMAIL_SENDER)

if not EMAIL_SENDER or not EMAIL_PASSWORD:
    print('Missing sender or password')
    raise SystemExit(1)

try:
    context = ssl.create_default_context()
    with smtplib.SMTP(EMAIL_SMTP_SERVER, EMAIL_SMTP_PORT, timeout=20) as server:
        server.ehlo()
        server.starttls(context=context)
        server.ehlo()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
    print('SMTP login: SUCCESS')
except Exception as e:
    print('SMTP login: FAILED')
    print('Error:', e)
    raise SystemExit(2)
