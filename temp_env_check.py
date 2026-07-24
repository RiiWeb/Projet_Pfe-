import os
from pathlib import Path
BASE_DIR = Path('c:/pfe')
env_path = BASE_DIR / '.env'
print('exists', env_path.exists())
text = env_path.read_text(encoding='utf-8')
print('text=', repr(text))
for line in text.splitlines():
    line=line.strip()
    if not line or line.startswith('#') or '=' not in line:
        continue
    key,value=line.split('=',1)
    key=key.strip(); value=value.strip().strip('"').strip("'")
    if key and value and key not in os.environ:
        os.environ[key]=value
print('ADMIN_NOTIFICATION_EMAILS=', os.environ.get('ADMIN_NOTIFICATION_EMAILS'))
print('EMAIL_SENDER=', os.environ.get('EMAIL_SENDER'))
print('EMAIL_PASSWORD=', os.environ.get('EMAIL_PASSWORD'))
print('EMAIL_ENABLED=', os.environ.get('EMAIL_ENABLED'))
print('list=', [e.strip() for e in os.environ.get('ADMIN_NOTIFICATION_EMAILS', os.environ.get('EMAIL_SENDER','')).split(',') if e.strip()])
