from flask import Flask, request, jsonify, render_template, redirect, url_for, send_file, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from datetime import datetime
import os
from pathlib import Path
import ssl
import threading
import torch, torch.nn as nn, numpy as np, pickle
import asyncio, edge_tts, tempfile, json, sqlite3
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

BASE_DIR = Path(__file__).resolve().parent

def load_env_file(env_path):
    if not env_path.is_file():
        return
    with env_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and value and (key not in os.environ or not os.environ.get(key)):
                os.environ[key] = value

load_env_file(BASE_DIR / '.env')

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///fraud.db'
app.config['SECRET_KEY'] = 'fraudintel-secret-2025'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SECURE'] = False
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['PERMANENT_SESSION_LIFETIME'] = 3600


# Email Configuration 
EMAIL_ENABLED      = os.environ.get('EMAIL_ENABLED', 'true').lower() in ('1','true','yes')
EMAIL_SENDER       = os.environ.get('EMAIL_SENDER', 'souai.rayen.rs@gmail.com')
EMAIL_PASSWORD     = os.environ.get('EMAIL_PASSWORD', '')
EMAIL_SMTP_SERVER  = os.environ.get('EMAIL_SMTP_SERVER', 'smtp.gmail.com')
EMAIL_SMTP_PORT    = int(os.environ.get('EMAIL_SMTP_PORT', 587))
EMAIL_CONFIGURED   = EMAIL_ENABLED and bool(EMAIL_SENDER) and bool(EMAIL_PASSWORD)
if EMAIL_ENABLED and not EMAIL_CONFIGURED:
    print('WARNING: Email is enabled but SMTP credentials are incomplete. Email notifications are disabled.')


ADMIN_NOTIFICATION_EMAILS = [e.strip() for e in os.environ.get('ADMIN_NOTIFICATION_EMAILS', EMAIL_SENDER).split(',') if e.strip()]
print(f'DEBUG EMAIL_CONFIGURED={EMAIL_CONFIGURED} EMAIL_SENDER={EMAIL_SENDER} ADMIN_NOTIFICATION_EMAILS={ADMIN_NOTIFICATION_EMAILS}')

db            = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view    = 'login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.session_protection = 'strong'

BASE_DIR        = Path(__file__).resolve().parent
HISTORY_DB_PATH = BASE_DIR / 'history.db'


# DATABASE MODELS

class User(UserMixin, db.Model):
    id        = db.Column(db.Integer, primary_key=True)
    username  = db.Column(db.String(100), unique=True, nullable=False)
    password  = db.Column(db.String(200), nullable=False)
    role      = db.Column(db.String(20), default='analyst')  # admin / analyst / agent
    is_admin  = db.Column(db.Boolean, default=False)
    bank_name = db.Column(db.String(100), nullable=True)

    def set_password(self, pwd):  self.password = generate_password_hash(pwd)
    def check_password(self, pwd): return check_password_hash(self.password, pwd)

class Transaction(db.Model):
    id           = db.Column(db.Integer, primary_key=True)
    user_id      = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    amount       = db.Column(db.Float)
    age          = db.Column(db.Float)
    income       = db.Column(db.Float)
    debt         = db.Column(db.Float)
    credit_score = db.Column(db.Float)
    hour         = db.Column(db.Integer)
    day          = db.Column(db.Integer)
    month        = db.Column(db.Integer)
    year         = db.Column(db.Integer)
    verdict      = db.Column(db.String(20))
    probability  = db.Column(db.Float)
    model_used   = db.Column(db.String(50))
    created_at   = db.Column(db.DateTime, default=datetime.utcnow)

class AgentTransaction(db.Model):
    id             = db.Column(db.Integer, primary_key=True)
    agent_id       = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    transaction_id = db.Column(db.String(100))
    bank_name      = db.Column(db.String(100))
    amount         = db.Column(db.Float)
    age            = db.Column(db.Float)
    income         = db.Column(db.Float)
    debt           = db.Column(db.Float)
    credit_score   = db.Column(db.Float)
    hour           = db.Column(db.Integer)
    day            = db.Column(db.Integer)
    month          = db.Column(db.Integer)
    year           = db.Column(db.Integer)
    verdict        = db.Column(db.String(20))
    probability    = db.Column(db.Float)
    model_used     = db.Column(db.String(50))
    created_at     = db.Column(db.DateTime, default=datetime.utcnow)

class Notification(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    message    = db.Column(db.String(500))
    is_read    = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class ContactRequest(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    full_name  = db.Column(db.String(100), nullable=False)
    email      = db.Column(db.String(100), nullable=False)
    message    = db.Column(db.Text, nullable=False)
    is_read    = db.Column(db.Boolean, default=False)
    account_created = db.Column(db.Boolean, default=False)
    created_username = db.Column(db.String(100), nullable=True)
    notification_sent = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    #feedback 

class Feedback(db.Model):
    id          = db.Column(db.Integer, primary_key=True)
    user_id     = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)  
    full_name   = db.Column(db.String(100), nullable=False)
    email       = db.Column(db.String(100), nullable=False)
    subject     = db.Column(db.String(200), nullable=False)
    category    = db.Column(db.String(50), nullable=False)  # fraud, technical, general
    message     = db.Column(db.Text, nullable=False)
    priority    = db.Column(db.String(20), default='normal')  # low, normal, high, urgent
    status      = db.Column(db.String(20), default='pending')  # pending, in_review, resolved, closed
    is_read     = db.Column(db.Boolean, default=False)
    response    = db.Column(db.Text, nullable=True)
    responded_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    responded_at = db.Column(db.DateTime, nullable=True)
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    try:    return db.session.get(User, int(user_id))
    except: return None

@login_manager.unauthorized_handler
def unauthorized(): return redirect(url_for('login'))


# email sending


def send_email(to_email, subject, body_html):
    """Send email notification to user in a background thread."""
    if not EMAIL_CONFIGURED:
        print(f"Email service unavailable: sending to {to_email} skipped.")
        return {'ok': False, 'error': 'email_disabled'}
        
    def _send():
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = EMAIL_SENDER
            msg['To'] = to_email
            msg.attach(MIMEText(body_html, 'html', 'utf-8'))

            context = ssl.create_default_context()
            with smtplib.SMTP(EMAIL_SMTP_SERVER, EMAIL_SMTP_PORT, timeout=20) as server:
                server.ehlo()
                server.starttls(context=context)
                server.ehlo()
                server.login(EMAIL_SENDER, EMAIL_PASSWORD)
                server.send_message(msg)
        except Exception as e:
            print(f"SMTP Background Error to {to_email}: {e}")

    # start email process
    t = threading.Thread(target=_send)
    t.start()
    
    # return success
    return {'ok': True, 'error': None}


def build_email_header(title, subtitle):
    logo_url = os.environ.get('EMAIL_LOGO_URL')
    if logo_url:
        logo_html = f"<img src=\"{logo_url}\" alt=\"BlockSafe logo\" style=\"width:48px;height:48px;border-radius:12px;object-fit:cover;display:block\"/>"
    else:
        logo_html = '''
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="display:block;filter:drop-shadow(0 2px 6px rgba(0,0,0,0.2))">
                <defs>
                    <linearGradient id="logoGradEmail" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stop-color="#1a56db"/>
                        <stop offset="100%" stop-color="#0ea5e9"/>
                    </linearGradient>
                </defs>
                <path d="M12 2L4 6.5V14C4 18.5 7.5 22 12 22C16.5 22 20 18.5 20 14V6.5L12 2Z" fill="url(#logoGradEmail)"/>
                <path d="M8.5 12L11 14.5L15.5 10" stroke="#fff" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
        '''

    return f"""
        <div style="background:#0f172a;padding:22px 24px;border-radius:16px 16px 0 0;color:#fff;">
            <div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap">
                {logo_html}
                <div>
                    <div style="font-size:20px;font-weight:700;letter-spacing:-0.03em;">BlockSafe</div>
                    <div style="font-size:13px;opacity:.76;line-height:1.4;max-width:440px;">{subtitle}</div>
                </div>
            </div>
            <div style="margin-top:18px;font-size:24px;font-weight:700;line-height:1.1;">{title}</div>
        </div>
    """

#db migrations

_initialized = False

def run_migrations():
    """Run database migrations"""
    try:
        db.create_all()
        # Safe migrations - add columns if they don't exist
        migration_sql = [
            "ALTER TABLE user ADD COLUMN role VARCHAR(20) DEFAULT 'analyst'",
            "ALTER TABLE user ADD COLUMN bank_name VARCHAR(100)",
            "ALTER TABLE user ADD COLUMN is_admin BOOLEAN DEFAULT 0",
            # ContactRequest migrations
            "ALTER TABLE contact_request ADD COLUMN account_created BOOLEAN DEFAULT 0",
            "ALTER TABLE contact_request ADD COLUMN created_username VARCHAR(100)",
            "ALTER TABLE contact_request ADD COLUMN created_password VARCHAR(100)",
            "ALTER TABLE contact_request ADD COLUMN notification_sent BOOLEAN DEFAULT 0",
        ]
        for sql in migration_sql:
            try:
                db.session.execute(db.text(sql))
                db.session.commit()
            except Exception as e:
                db.session.rollback()
                # Column likely already exists
                pass
    except Exception as e:
        print(f"Migration error: {e}")

@app.before_request
def initialize():
    global _initialized
    if _initialized: return
    run_migrations()
    init_history_db()
    
    # Default users
    defaults = [
        ('analyst', 'fraudintel2025', 'analyst', False, None),
        ('admin',   'admin2025',      'admin',   True,  None),
        ('agent',   'agent2025',      'agent',   False, 'Demo Bank'),
    ]
    
    for uname, pwd, role, is_admin, bank in defaults:
        try:
            # Check if user already exists in the database
            if not User.query.filter_by(username=uname).first():
                u = User(username=uname, role=role, is_admin=is_admin, bank_name=bank)
                u.set_password(pwd)
                db.session.add(u)
                db.session.commit()  # Cleanly commit immediately to establish the row
        except Exception:
            db.session.rollback()  # Handle concurrent insertion gracefully if another thread wins
            
    _initialized = True

#history

def get_history_connection():
    conn = sqlite3.connect(HISTORY_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_history_db():
    with get_history_connection() as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER, model_name TEXT, input_data TEXT,
            probability REAL, risk_level TEXT, verdict TEXT,
            confidence REAL, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')
        conn.commit()

def save_history(user_id, model_name, input_data, probability, risk_level, verdict, confidence):
    with get_history_connection() as conn:
        conn.execute('''INSERT INTO history
            (user_id,model_name,input_data,probability,risk_level,verdict,confidence)
            VALUES (?,?,?,?,?,?,?)''',
            (user_id, model_name, json.dumps(input_data), probability, risk_level, verdict, confidence))
        conn.commit()

def get_user_history(user_id):
    with get_history_connection() as conn:
        rows = conn.execute(
            'SELECT * FROM history WHERE user_id=? ORDER BY created_at DESC, id DESC', (user_id,)).fetchall()
    history = []
    for row in rows:
        row_dict = dict(row)
        input_data = row_dict.get('input_data')
        if isinstance(input_data, str):
            try:
                row_dict['input_data'] = json.loads(input_data)
            except json.JSONDecodeError:
                row_dict['input_data'] = {}
        history.append(row_dict)
    return history

def get_risk_level(p):
    if p >= 70: return 'HIGH RISK'
    if p >= 40: return 'MODERATE RISK'
    return 'LOW RISK'

#models

class FTTransformer(nn.Module):
    def __init__(self, num_numeric, num_locations, num_devices, embed_dim=8):
        super().__init__()
        self.location_emb = nn.Embedding(num_locations, embed_dim)
        self.device_emb   = nn.Embedding(num_devices,   embed_dim)
        self.model = nn.Sequential(
            nn.Linear(num_numeric + embed_dim + embed_dim, 128), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Linear(64, 32),  nn.ReLU(), nn.Linear(32, 1)
        )
    def forward(self, x_num, x_cat):
        loc = self.location_emb(x_cat[:,0].long())
        dev = self.device_emb(x_cat[:,1].long())
        return self.model(torch.cat((x_num, loc, dev), dim=1))

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim,64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.4),
            nn.Linear(64,32),        nn.ReLU(), nn.BatchNorm1d(32), nn.Dropout(0.4),
            nn.Linear(32,1)
        )
    def forward(self, x): return self.model(x)

device = torch.device('cpu')

with open('fttransformer.pkl','rb') as f:     ft_bundle     = pickle.load(f)
with open('fttransformer_extracted.pkl','rb') as f: ft_ext_bundle = pickle.load(f)
with open('mlp.pkl','rb') as f:               mlp_bundle    = pickle.load(f)
with open('mlp_extracted.pkl','rb') as f:     mlp_ext_bundle= pickle.load(f)

ft_model      = ft_bundle['model'].to(device);     ft_model.eval()
ft_ext_model  = ft_ext_bundle['model'].to(device); ft_ext_model.eval()
mlp_model     = mlp_bundle['model'].to(device);    mlp_model.eval()
mlp_ext_model = mlp_ext_bundle['model'].to(device);mlp_ext_model.eval()
ft_ext_scaler = ft_ext_bundle['scaler']
mlp_scaler    = mlp_bundle['scaler']
mlp_ext_scaler= mlp_ext_bundle['scaler']

from sklearn.preprocessing import MinMaxScaler
import pandas as _pd
_ft_cols = ['amount','age','income','debt','credit_score','hour','day','month','year']
_ft_min  = [1.0,   18.0,  20000.0,  0.0,    300.0, 0.0, 1.0, 1.0, 2020.0]
_ft_max  = [4999.0, 70.0, 149999.0, 49999.0, 850.0, 23.0, 31.0, 12.0, 2024.0]
ft_scaler = MinMaxScaler()
ft_scaler.fit(_pd.DataFrame([_ft_min, _ft_max], columns=_ft_cols))

DATASET_YEAR_MIN = 2023
DATASET_YEAR_MAX = 2024

VOICE = 'en-US-JennyNeural'



def admin_only(f):
    @wraps(f)
    def decorated(*a, **kw):
        if not current_user.is_authenticated or not current_user.is_admin:
            return jsonify({'error':'Admin access required'}), 403
        return f(*a, **kw)
    return decorated

def agent_only(f):
    @wraps(f)
    def decorated(*a, **kw):
        if not current_user.is_authenticated or current_user.role != 'agent':
            return redirect(url_for('login'))
        return f(*a, **kw)
    return decorated


def run_prediction(data):
    model_type = data.get('model_type','ft_original')
    
    try:
        amount       = float(data.get('amount', 0))
        age          = float(data.get('age', 0))
        income       = float(data.get('income', 0))
        debt         = float(data.get('debt', 0))
        credit_score = float(data.get('credit_score', 0))
        hour         = float(data.get('hour', 0))
        day          = float(data.get('day', 0))
        month        = float(data.get('month', 0))
        year         = float(data.get('year', 0))
    except (ValueError, TypeError):
      
        return {
            'verdict': 'ERROR', 'probability': 0.0, 'confidence': 0.0,
            'risk_level': 'INVALID INPUT', 'model_used': 'N/A',
            'amount': 0, 'age': 0, 'income': 0, 'debt': 0,
            'credit_score': 0, 'hour': 0, 'day': 0, 'month': 0, 'year': 0,
            'error_msg': 'Invalid numerical input provided.'
        }

    dti = debt/(income+1e-6); ati = amount/(income+1e-6)
    atd = amount/(debt+1e-6); cxa = credit_score*amount; axd = age*debt


    try:
        tx_date = datetime(int(year), int(month), int(day)).date()
    except ValueError:
        return {
            'verdict': 'ERROR', 'probability': 0.0, 'confidence': 0.0,
            'risk_level': 'INVALID DATE', 'model_used': 'N/A',
            'amount': amount, 'age': age, 'income': income, 'debt': debt,
            'credit_score': credit_score, 'hour': int(hour), 'day': int(day),
            'month': int(month), 'year': int(year), 'error_msg': 'Invalid transaction date.'
        }

    today = datetime.utcnow().date()
    dataset_min = datetime(DATASET_YEAR_MIN, 1, 1).date()
    dataset_max = datetime(DATASET_YEAR_MAX, 12, 31).date()
    is_dataset_date = dataset_min <= tx_date <= dataset_max
    is_today = tx_date == today
    if not (is_dataset_date or is_today):
        return {
            'verdict': 'ERROR', 'probability': 0.0, 'confidence': 0.0,
            'risk_level': 'INVALID DATE', 'model_used': 'N/A',
            'amount': amount, 'age': age, 'income': income, 'debt': debt,
            'credit_score': credit_score, 'hour': int(hour), 'day': int(day),
            'month': int(month), 'year': int(year),
            'error_msg': f'Date must be within {DATASET_YEAR_MIN}-{DATASET_YEAR_MAX} or today.'
        }

    arr9  = np.array([[amount,age,income,debt,credit_score,hour,day,month,year]])
    arr14 = np.array([[amount,age,income,debt,credit_score,hour,day,month,year,dti,ati,atd,cxa,axd]])

    
    
    if model_type == 'ft_original':
        scaled = ft_scaler.transform(arr9)
        x_num = torch.tensor(scaled, dtype=torch.float32)
        x_cat = torch.tensor([[0,0]], dtype=torch.long)
        with torch.no_grad(): out = torch.sigmoid(ft_model(x_num, x_cat))
        label = 'FTTransformer — Original'
    elif model_type == 'ft_extracted':
        scaled = ft_ext_scaler.transform(arr14)
        x_num = torch.tensor(scaled, dtype=torch.float32)
        x_cat = torch.tensor([[0,0]], dtype=torch.long)
        with torch.no_grad(): out = torch.sigmoid(ft_ext_model(x_num, x_cat))
        label = 'FTTransformer — Extracted'
    elif model_type == 'mlp_original':
        scaled = mlp_scaler.transform(arr9)
        x = torch.tensor(scaled, dtype=torch.float32)
        with torch.no_grad(): out = torch.sigmoid(mlp_model(x))
        label = 'MLP — Original'
    else:
        scaled = mlp_ext_scaler.transform(arr14)
        x = torch.tensor(scaled, dtype=torch.float32)
        with torch.no_grad(): out = torch.sigmoid(mlp_ext_model(x))
        label = 'MLP — Extracted'

    prob    = out.item()
    verdict = 'FRAUD' if prob >= 0.5 else 'LEGITIMATE'
    conf    = prob if verdict=='FRAUD' else 1-prob
    prob_pct = round(prob*100, 2)
    conf_pct = round(conf*100, 2)
    
    return {
        'verdict':verdict, 'probability':prob_pct, 'confidence':conf_pct,
        'risk_level':get_risk_level(prob_pct), 'model_used':label,
        'amount':amount, 'age':age, 'income':income, 'debt':debt,
        'credit_score':credit_score, 'hour':int(hour), 'day':int(day),
        'month':int(month), 'year':int(year)
    }
#auth_routes

@app.route('/landing')
def landing():
    return render_template('landing.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if current_user.is_authenticated:
        if current_user.role == 'admin':  return redirect(url_for('admin_dashboard'))
        if current_user.role == 'agent':  return redirect(url_for('agent_dashboard'))
        return redirect(url_for('index'))
    error = None
    if request.method == 'POST':
        u = User.query.filter_by(username=request.form.get('username','').strip()).first()
        if u and u.check_password(request.form.get('password','')):
            login_user(u, remember=False)
            if u.role == 'admin': return redirect(url_for('admin_dashboard'))
            if u.role == 'agent': return redirect(url_for('agent_dashboard'))
            return redirect(url_for('index'))
        error = 'Invalid username or password.'
    return render_template('login.html', error=error)

@app.route('/logout')
@login_required
def logout():
    logout_user(); session.clear()
    return redirect(url_for('login'))

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        try:
            full_name = request.form.get('full_name', '').strip()
            email = request.form.get('email', '').strip()
            message = request.form.get('message', '').strip()
            
            if not all([full_name, email, message]):
                return render_template('contact.html', error='Please fill in all fields'), 400
            
            contact_req = ContactRequest(full_name=full_name, email=email, message=message)
            db.session.add(contact_req)
            db.session.commit()
            
            # Send confirmation email to user
            confirmation_email = f"""
            <html><body style="font-family:Arial,sans-serif;color:#1a2744;background:#eef2f7;padding:0;margin:0;">
                <div style="max-width:700px;margin:0 auto;overflow:hidden;border-radius:20px;box-shadow:0 28px 100px rgba(15,23,42,0.12);background:#ffffff">
                    {build_email_header('Contact Request Received', 'Your message has been received by the BlockSafe support team.')}
                    <div style="padding:28px 32px 36px;">
                        <p style="margin:0 0 18px;font-size:15px;line-height:1.8;color:#0f172a">Hi {full_name},</p>
                        <p style="margin:0 0 20px;font-size:15px;line-height:1.8;color:#334155">Thank you for reaching out. Your contact request has been received and is currently being reviewed by our security operations team.</p>
                        <div style="background:#f8fafc;padding:18px;border-radius:14px;margin:18px 0;border-left:4px solid #1a56db;">
                            <p style="margin:0 0 10px;font-weight:700;color:#0f172a">Next steps</p>
                            <ul style="margin:0;padding-left:20px;color:#475569;line-height:1.8">
                                <li>Our team is reviewing your request.</li>
                                <li>We will follow up with any additional questions.</li>
                                <li>If approved, account setup details will be sent to you.</li>
                            </ul>
                        </div>
                        <p style="margin:0 0 12px;font-size:13px;color:#64748b">Request ID: <strong>#{contact_req.id}</strong></p>
                        <p style="margin:0;font-size:13px;color:#64748b">If you need to update your request, reply to this message or visit <a href="http://localhost:5000/check-account-status" style="color:#1a56db;text-decoration:none">Check Account Status</a>.</p>
                    </div>
                </div>
            </body></html>
            """
            # Send confirmation email to user in background so contact submit is not blocked by email latency.
            send_res = {'ok': False, 'error': 'email_disabled'}
            if EMAIL_CONFIGURED:
                threading.Thread(target=send_email, args=(email, 'BlockSafe Contact Request Confirmation', confirmation_email), daemon=True).start()

            admin_body = f"""
            <html><body style="font-family:Arial,sans-serif;color:#1a2744;background:#eef2f7;padding:0;margin:0;">
                <div style="max-width:700px;margin:0 auto;overflow:hidden;border-radius:20px;box-shadow:0 28px 100px rgba(15,23,42,0.12);background:#ffffff">
                    {build_email_header('New Contact Request Submitted', 'A new user contact request has been received.')}
                    <div style="padding:28px 32px 36px;">
                        <p style="margin:0 0 18px;font-size:15px;line-height:1.8;color:#0f172a">A new contact request was submitted by {full_name}.</p>
                        <p style="margin:0 0 12px;font-size:15px;color:#334155"><strong>Email:</strong> {email}</p>
                        <p style="margin:0 0 20px;font-size:15px;color:#334155"><strong>Message:</strong></p>
                        <div style="background:#f8fafc;padding:18px;border-radius:14px;margin:18px 0;border-left:4px solid #1a56db;color:#475569;line-height:1.8">{message}</div>
                        <p style="margin:0;font-size:13px;color:#64748b">Request ID: <strong>#{contact_req.id}</strong></p>
                        <p style="margin:0;font-size:13px;color:#64748b">Review this request in the admin portal or by searching the request ID.</p>
                    </div>
                </div>
            </body></html>
            """
            if EMAIL_CONFIGURED:
                threading.Thread(target=lambda: [send_email(admin_email, f'New Contact Request: {full_name}', admin_body) for admin_email in ADMIN_NOTIFICATION_EMAILS], daemon=True).start()

            return render_template('contact.html', success=True)
        except Exception as e:
            db.session.rollback()
            return render_template('contact.html', error='An error occurred. Please try again.'), 500
    
    return render_template('contact.html')

@app.route('/feedback', methods=['GET', 'POST'])
@login_required
def feedback():
    if current_user.role not in ('admin', 'analyst', 'agent'):
        return render_template('feedback.html', error='Feedback is available only to registered users.'), 403

    if request.method == 'POST':
        try:
            # Debug: log incoming form data for troubleshooting
            try:
                form_data = {k: v for k, v in request.form.items()}
            except Exception:
                form_data = {}
            print(f"DEBUG /feedback form received: {form_data}")

            full_name = request.form.get('full_name', '').strip()
            email = request.form.get('email', '').strip()
            subject = request.form.get('subject', '').strip()
            category = request.form.get('category', 'general')
            message = request.form.get('message', '').strip()

            missing = [name for name, val in (('full_name', full_name), ('email', email), ('subject', subject), ('message', message)) if not val]
            if missing:
                err = f"Please fill in all fields: missing {', '.join(missing)}"
                print(f"DEBUG /feedback missing fields: {missing}")
                return render_template('feedback.html', error=err), 400

            # Determine priority based on category
            priority = 'normal'
            if category == 'fraud':
                priority = 'high'
            elif category == 'technical':
                priority = 'normal'
            elif category == 'urgent':
                priority = 'urgent'

            feedback_entry = Feedback(
                user_id=current_user.id if current_user.is_authenticated else None,
                full_name=full_name,
                email=email,
                subject=subject,
                category=category,
                message=message,
                priority=priority
            )
            db.session.add(feedback_entry)
            db.session.commit()

            # Send confirmation email to user
            confirmation_email = f"""
            <html><body style="font-family:Arial,sans-serif;color:#1a2744;background:#eef2f7;padding:0;margin:0;">
                <div style="max-width:700px;margin:0 auto;overflow:hidden;border-radius:20px;box-shadow:0 28px 100px rgba(15,23,42,0.12);background:#ffffff">
                    {build_email_header('Feedback Received', 'Your report has been securely logged and is now in review.')}
                    <div style="padding:28px 32px 36px;">
                        <p style="margin:0 0 18px;font-size:15px;line-height:1.8;color:#0f172a">Hi {full_name},</p>
                        <p style="margin:0 0 20px;font-size:15px;line-height:1.8;color:#334155">Thank you for sharing your feedback on <strong>{subject}</strong>. Your input is important to our fraud intelligence and product experience teams.</p>
                        <div style="background:#f8fafc;padding:18px;border-radius:14px;margin:18px 0;border-left:4px solid #1a56db;">
                            <p style="margin:0 0 10px;font-weight:700;color:#0f172a">Feedback details</p>
                            <p style="margin:0;color:#475569;line-height:1.8"><strong>Category:</strong> {category.title()}</p>
                            <p style="margin:0;color:#475569;line-height:1.8"><strong>Priority:</strong> {priority.title()}</p>
                            <p style="margin:0;color:#475569;line-height:1.8"><strong>Reference ID:</strong> #{feedback_entry.id}</p>
                        </div>
                        <p style="margin:0;font-size:15px;line-height:1.8;color:#334155">Our specialist team will review this and get back to you if additional information is needed.</p>
                    </div>
                </div>
            </body></html>
            """
            user_send = send_email(email, f'BlockSafe Feedback Received - {subject}', confirmation_email)
            print(f"DEBUG /feedback user_send result: {user_send}")

            # Notify admins
            admin_results = []
            admin_body = f"""
            <html><body style=\"font-family:Arial,sans-serif;color:#1a2744;background:#eef2f7;padding:0;margin:0;\">                
                <div style=\"max-width:700px;margin:0 auto;overflow:hidden;border-radius:20px;box-shadow:0 28px 100px rgba(15,23,42,0.12);background:#ffffff\">                    
                    {build_email_header('New Feedback Submitted', 'A new feedback item has been logged in the BlockSafe portal.')}
                    <div style=\"padding:28px 32px 36px;\">                        
                        <p style=\"margin:0 0 16px;font-size:15px;line-height:1.8;color:#334155\">A new feedback submission has been received and requires review.</p>
                        <div style=\"background:#f8fafc;padding:18px;border-radius:14px;margin:18px 0;border-left:4px solid #1a56db;\">                            
                            <p style=\"margin:0 0 8px;font-weight:700;color:#0f172a\">Submission details</p>
                            <p style=\"margin:0;color:#475569;line-height:1.8\"><strong>From:</strong> {full_name} &lt;{email}&gt;</p>
                            <p style=\"margin:0;color:#475569;line-height:1.8\"><strong>Subject:</strong> {subject}</p>
                            <p style=\"margin:0;color:#475569;line-height:1.8\"><strong>Category:</strong> {category.title()}</p>
                            <p style=\"margin:0;color:#475569;line-height:1.8\"><strong>Reference ID:</strong> #{feedback_entry.id}</p>
                        </div>
                        <div style=\"background:white;padding:18px;border-radius:14px;border:1px solid #e2e8f0;color:#475569;line-height:1.8;white-space:pre-line\">{message}</div>
                    </div>
                </div>
            </body></html>
            """
            for admin_email in ADMIN_NOTIFICATION_EMAILS:
                res = send_email(admin_email, f'New Feedback: {subject}', admin_body)
                admin_results.append({'email': admin_email, 'result': res})
                print(f"DEBUG /feedback admin_notify {admin_email} -> {res}")

            # If user email failed, surface error. If admin notifications failed, surface but still accept feedback.
            email_error = None
            if not user_send.get('ok') and user_send.get('error') != 'email_disabled':
                email_error = f"user_email:{user_send.get('error')}"
            admin_errors = [r for r in admin_results if not r['result'].get('ok') and r['result'].get('error') != 'email_disabled']

            if email_error or admin_errors:
                return render_template('feedback.html', success=True, email_error=email_error, admin_errors=admin_errors)

            return render_template('feedback.html', success=True)
        except Exception as e:
            db.session.rollback()
            return render_template('feedback.html', error='An error occurred. Please try again.'), 500

    return render_template('feedback.html')
@app.route('/admin/feedback/<int:id>/respond', methods=['POST'])
@login_required
def admin_feedback_respond(id):
    # Check if the user is an admin
    if current_user.role != 'admin': 
        return jsonify({'error': 'Forbidden'}), 403
        
    fb = Feedback.query.get_or_404(id)
    data = request.json
    response_text = data.get('response', '').strip()
    
    if not response_text:
        return jsonify({'error': 'Response text is required'}), 400
        
    # Update the database
    fb.status = 'resolved'
    fb.response = response_text
    db.session.commit()
    
    # Send the response email to the user
    if fb.email:
        email_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; padding: 20px;">
            <h2>BlockSafe Support Response</h2>
            <p><strong>Regarding your feedback:</strong><br/> "{fb.message}"</p>
            <hr/>
            <p><strong>Our Response:</strong><br/> {response_text}</p>
            <br/>
            <p>Best regards,<br/>The BlockSafe Team</p>
        </body>
        </html>
        """
        send_email(fb.email, "BlockSafe - Response to your Feedback", email_body)
        
    return jsonify({'success': True})

#analyst_rules

@app.route('/')
def index_root():
    if not current_user.is_authenticated:
        return render_template('landing.html')
    if current_user.role == 'admin': return redirect(url_for('admin_dashboard'))
    if current_user.role == 'agent': return redirect(url_for('agent_dashboard'))
    return render_template('index.html', username=current_user.username, is_admin=current_user.is_admin)

@app.route('/dashboard')
@login_required
def index():
    if current_user.role == 'admin': return redirect(url_for('admin_dashboard'))
    if current_user.role == 'agent': return redirect(url_for('agent_dashboard'))
    return render_template('index.html', username=current_user.username, is_admin=current_user.is_admin)

@app.route('/speak')
@login_required
def speak_route():
    text = request.args.get('text','')
    if not text: return jsonify({'error':'no text'}), 400
    tmp = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
    tmp.close()
    async def _gen(): await edge_tts.Communicate(text=text, voice=VOICE).save(tmp.name)
    asyncio.run(_gen())
    return send_file(tmp.name, mimetype='audio/mpeg')

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        result = run_prediction(request.get_json())
        tx = Transaction(user_id=current_user.id, amount=result['amount'], age=result['age'],
            income=result['income'], debt=result['debt'], credit_score=result['credit_score'],
            hour=result['hour'], day=result['day'], month=result['month'], year=result['year'],
            verdict=result['verdict'], probability=result['probability'], model_used=result['model_used'])
        db.session.add(tx); db.session.commit()
        save_history(current_user.id, result['model_used'],
            {k:result[k] for k in ['amount','age','income','debt','credit_score','hour','day','month','year']},
            result['probability'], result['risk_level'], result['verdict'], result['confidence'])
        return jsonify(result)
    except Exception as e: return jsonify({'error':str(e)}), 400

@app.route('/history')
@login_required
def history():
    return jsonify({'history': get_user_history(current_user.id)})

@app.route('/profile')
@login_required
def profile():
    txs = Transaction.query.filter_by(user_id=current_user.id).all()
    return jsonify({'username':current_user.username,'role':current_user.role,'is_admin':current_user.is_admin,
        'tx_count':len(txs),'fraud_count':sum(1 for t in txs if t.verdict=='FRAUD'),
        'legit_count':sum(1 for t in txs if t.verdict=='LEGITIMATE')})

#admin_routes

@app.route('/admin')
@login_required
@admin_only
def admin_dashboard():
    return render_template('admin.html', username=current_user.username)

@app.route('/admin/stats')
@login_required
@admin_only
def admin_stats():
    # Include both analyst and agent transactions
    a_total = Transaction.query.count()
    a_fraud = Transaction.query.filter_by(verdict='FRAUD').count()
    a_legit = Transaction.query.filter_by(verdict='LEGITIMATE').count()
    b_total = AgentTransaction.query.count()
    b_fraud = AgentTransaction.query.filter_by(verdict='FRAUD').count()
    b_legit = AgentTransaction.query.filter_by(verdict='LEGITIMATE').count()
    total = a_total + b_total
    fraud = a_fraud + b_fraud
    legit = a_legit + b_legit
    users    = User.query.count()
    agents   = User.query.filter_by(role='agent').count()
    analysts = User.query.filter_by(role='analyst').count()
    mc = {}
    for tx in Transaction.query.all():
        mc[tx.model_used] = mc.get(tx.model_used,0)+1
    for tx in AgentTransaction.query.all():
        mc[tx.model_used] = mc.get(tx.model_used,0)+1
    return jsonify({'total_tx':total,'fraud_tx':fraud,'legit_tx':legit,
        'fraud_rate':round(fraud/total*100,1) if total else 0,
        'total_users':users,'agents':agents,'analysts':analysts,'model_counts':mc})

@app.route('/admin/users')
@login_required
@admin_only
def admin_users():
    return jsonify({'users':[{
        'id':u.id,'username':u.username,'role':u.role,'is_admin':u.is_admin,
        'bank_name':u.bank_name
    } for u in User.query.all()]})

@app.route('/admin/create-user', methods=['POST'])
@login_required
@admin_only
def create_user():
    data = request.get_json()
    if User.query.filter_by(username=data['username']).first():
        return jsonify({'error':'Username already exists'}), 400
    u = User(username=data['username'], role=data.get('role','analyst'),
             is_admin=data.get('role')=='admin', bank_name=data.get('bank_name'))
    u.set_password(data['password'])
    db.session.add(u); db.session.commit()
    return jsonify({'success':True,'id':u.id})

@app.route('/admin/delete-user/<int:uid>', methods=['DELETE'])
@login_required
@admin_only
def delete_user(uid):
    u = db.session.get(User, uid)
    if not u: return jsonify({'error':'User not found'}), 404
    Transaction.query.filter_by(user_id=uid).delete()
    db.session.delete(u); db.session.commit()
    return jsonify({'success':True})

@app.route('/admin/all-history')
@login_required
@admin_only
def admin_all_history():
    # Analyst transactions
    rows = []
    for t in Transaction.query.order_by(Transaction.id.desc()).all():
        u = db.session.get(User, t.user_id)
        rows.append({
            'id':t.id, 'username':u.username if u else '?', 'role':'analyst',
            'transaction_id':'—', 'bank_name':'—',
            'amount':t.amount, 'verdict':t.verdict, 'probability':t.probability,
            'model_used':t.model_used, 'credit_score':t.credit_score,
            'created_at':t.created_at.strftime('%Y-%m-%d %H:%M') if t.created_at else ''
        })
    # Agent transactions
    for t in AgentTransaction.query.order_by(AgentTransaction.id.desc()).all():
        u = db.session.get(User, t.agent_id)
        rows.append({
            'id':f"A{t.id}", 'username':u.username if u else '?', 'role':'agent',
            'transaction_id':t.transaction_id or '—', 'bank_name':t.bank_name or '—',
            'amount':t.amount, 'verdict':t.verdict, 'probability':t.probability,
            'model_used':t.model_used, 'credit_score':t.credit_score,
            'created_at':t.created_at.strftime('%Y-%m-%d %H:%M') if t.created_at else ''
        })
    # Sort by date descending
    rows.sort(key=lambda x: x['created_at'], reverse=True)
    return jsonify({'history': rows[:200]})

@app.route('/admin/delete-transaction/<int:tid>', methods=['DELETE'])
@login_required
@admin_only
def delete_transaction(tid):
    t = db.session.get(Transaction, tid)
    if not t: return jsonify({'error':'Not found'}), 404
    db.session.delete(t); db.session.commit()
    return jsonify({'success':True})

@app.route('/admin/clear-history', methods=['DELETE'])
@login_required
@admin_only
def clear_history():
    n = Transaction.query.delete(); db.session.commit()
    with get_history_connection() as conn: conn.execute('DELETE FROM history'); conn.commit()
    return jsonify({'success':True,'records_deleted':n})

@app.route('/admin/contact-requests')
@login_required
@admin_only
def contact_requests():
    requests = ContactRequest.query.order_by(ContactRequest.created_at.desc()).all()
    contact_list = [{
        'id': r.id,
        'full_name': r.full_name,
        'email': r.email,
        'message': r.message,
        'is_read': r.is_read,
        'account_created': bool(r.account_created),
        'created_username': r.created_username,
        'notification_sent': bool(r.notification_sent),
        'created_at': r.created_at.strftime('%Y-%m-%d %H:%M:%S') if r.created_at else ''
    } for r in requests]
    return jsonify({'contact_requests': contact_list})

@app.route('/admin/contact-requests/<int:request_id>/mark-read', methods=['POST'])
@login_required
@admin_only
def mark_contact_read(request_id):
    req = db.session.get(ContactRequest, request_id)
    if req:
        req.is_read = True
        db.session.commit()
        return jsonify({'success': True})
    return jsonify({'error': 'Request not found'}), 404

@app.route('/admin/contact-requests/<int:request_id>/create-account', methods=['POST'])
@login_required
@admin_only
def create_account_from_contact(request_id):
    try:
        contact_req = db.session.get(ContactRequest, request_id)
        if not contact_req:
            return jsonify({'error': 'Contact request not found'}), 404
        
        data = request.get_json() or {}
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()
        role = data.get('role', 'analyst')
        
        if not username or not password:
            return jsonify({'error': 'Username and password required'}), 400
        
        if User.query.filter_by(username=username).first():
            return jsonify({'error': 'Username already exists'}), 400
        
        # Create the user
        user = User(username=username, role=role, bank_name=data.get('bank_name'))
        user.set_password(password)
        db.session.add(user)
        db.session.flush()  # Get the user ID
        
        # Update contact request
        contact_req.account_created = True
        contact_req.created_username = username
        contact_req.created_password = password
        contact_req.is_read = True
        
        # Send notification email (use header builder for consistent branding)
        email_body = f"""
        <html><body style="font-family:Arial,sans-serif;color:#1a2744;background:#eef2f7;padding:0;margin:0;">
            <div style="max-width:700px;margin:0 auto;overflow:hidden;border-radius:20px;box-shadow:0 28px 100px rgba(15,23,42,0.12);background:#ffffff">
                {build_email_header('Account Created', 'Your BlockSafe account has been created successfully.')}
                <div style="padding:28px 32px 36px;">
                    <p style="margin:0 0 18px;font-size:15px;line-height:1.8;color:#0f172a">Hi {contact_req.full_name},</p>
                    <p style="margin:0 0 20px;font-size:15px;line-height:1.8;color:#334155">Your BlockSafe account has been created and is ready to use.</p>
                    <div style="background:#f8fafc;padding:18px;border-radius:14px;margin:18px 0;border-left:4px solid #1a56db;">
                        <p style="margin:0 0 8px;font-weight:700;color:#0f172a">Account details</p>
                        <p style="margin:0;color:#475569;line-height:1.8"><strong>Username:</strong> {username}</p>
                        <p style="margin:0;color:#475569;line-height:1.8"><strong>Password:</strong> {password}</p>
                        <p style="margin:0;color:#475569;line-height:1.8"><strong>Role:</strong> {role}</p>
                    </div>
                    <p style="margin:0 0 12px;"><a href="http://localhost:5000/login" style="display:inline-block;background:#1a56db;color:white;padding:10px 20px;border-radius:8px;text-decoration:none">Sign In Now</a></p>
                    <p style="margin:18px 0 0;font-size:13px;color:#64748b">Please keep your credentials secure and change your password after first login.</p>
                </div>
            </div>
        </body></html>
        """
        if EMAIL_CONFIGURED:
            send_res = send_email(contact_req.email, 'BlockSafe Account Created', email_body)
        else:
            send_res = {'ok': True, 'error': None}
            print('DEBUG create_account_from_contact: email disabled, skipping notification email')
        print(f"DEBUG create_account_from_contact send_res: {send_res}")
        contact_req.notification_sent = bool(send_res.get('ok'))
        db.session.commit()
        if not send_res.get('ok') and send_res.get('error') != 'email_disabled':
            return jsonify({'success': True, 'username': username, 'email_error': send_res.get('error')}), 200
        return jsonify({'success': True, 'username': username}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Error creating account: {str(e)}'}), 500

@app.route('/admin/feedback')
@login_required
@admin_only
def admin_feedback():
    feedbacks = Feedback.query.order_by(Feedback.created_at.desc()).all()
    feedback_list = [{
        'id': f.id,
        'full_name': f.full_name,
        'email': f.email,
        'subject': f.subject,
        'category': f.category,
        'priority': f.priority,
        'status': f.status,
        'is_read': f.is_read,
        'created_at': f.created_at.strftime('%Y-%m-%d %H:%M') if f.created_at else ''
    } for f in feedbacks]
    return jsonify({'feedback': feedback_list})

@app.route('/admin/feedback/<int:feedback_id>/mark-read', methods=['POST'])
@login_required
@admin_only
def mark_feedback_read(feedback_id):
    feedback = db.session.get(Feedback, feedback_id)
    if feedback:
        feedback.is_read = True
        db.session.commit()
        return jsonify({'success': True})
    return jsonify({'error': 'Feedback not found'}), 404

@app.route('/admin/feedback/<int:feedback_id>/respond', methods=['POST'])
@login_required
@admin_only
def respond_to_feedback(feedback_id):
    try:
        feedback = db.session.get(Feedback, feedback_id)
        if not feedback:
            return jsonify({'error': 'Feedback not found'}), 404

        data = request.get_json()
        response = data.get('response', '').strip()

        if not response:
            return jsonify({'error': 'Response is required'}), 400

        feedback.response = response
        feedback.status = 'resolved'
        feedback.responded_by = current_user.id
        feedback.responded_at = datetime.utcnow()
        feedback.is_read = True

        db.session.commit()

        # Send response email to user
        response_email = f"""
        <html><body style="font-family:Arial,sans-serif;color:#1a2744;background:#eef2f7;padding:0;margin:0;">
            <div style="max-width:700px;margin:0 auto;overflow:hidden;border-radius:20px;box-shadow:0 28px 100px rgba(15,23,42,0.12);background:#ffffff">
                {build_email_header('Feedback Response Delivered', 'Your BlockSafe feedback has been reviewed and addressed.')}
                <div style="padding:28px 32px 36px;">
                    <p style="margin:0 0 18px;font-size:15px;line-height:1.8;color:#0f172a">Hi {feedback.full_name},</p>
                    <p style="margin:0 0 20px;font-size:15px;line-height:1.8;color:#334155">We’ve reviewed your feedback on <strong>{feedback.subject}</strong> and appreciate you taking the time to improve our platform.</p>
                    <div style="background:#f8fafc;padding:18px;border-radius:14px;margin:18px 0;border-left:4px solid #1a56db;">
                        <h3 style="margin:0 0 10px 0;color:#0f172a;font-size:15px;">Our response</h3>
                        <p style="margin:0;color:#475569;line-height:1.8;white-space:pre-line">{response}</p>
                    </div>
                    <p style="margin:0;font-size:15px;line-height:1.8;color:#334155">If you have any additional questions, please don’t hesitate to reply to this message.</p>
                    <p style="margin:20px 0 0;font-size:13px;color:#64748b">Feedback Reference: #{feedback.id}</p>
                </div>
            </div>
        </body></html>
        """
        send_email(feedback.email, f'Re: {feedback.subject}', response_email)

        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Error responding to feedback: {str(e)}'}), 500

@app.route('/check-account-status', methods=['GET', 'POST'])
def check_account_status():
    """Page where users can check if their account was created"""
    status = None
    email_searched = None
    
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        email_searched = email
        contact_req = ContactRequest.query.filter(
            db.func.lower(ContactRequest.email) == email
        ).first()
        
        if contact_req:
            if contact_req.account_created:
                status = {
                    'type': 'success',
                    'title': 'Account Ready!',
                    'message': f'Your account has been created. Username: <strong>{contact_req.created_username}</strong>',
                    'login_link': True
                }
            else:
                status = {
                    'type': 'pending',
                    'title': 'Request Under Review',
                    'message': 'Your request is being reviewed by our admin team. You will receive an email with your credentials once your account is approved.'
                }
        else:
            status = {
                'type': 'not_found',
                'title': 'No Request Found',
                'message': 'We could not find a contact request for this email. Please make sure you entered the correct email or <a href="/contact" style="color:#1a56db;text-decoration:none">submit a new request</a>.'
            }
    
    return render_template('check-status.html', status=status, email_searched=email_searched)


@app.route('/admin/email-test', methods=['POST'])
@login_required
@admin_only
def admin_email_test():
    """Send a test email to a specified address. JSON: {to, subject, body}"""
    data = request.get_json() or {}
    to = data.get('to')
    subject = data.get('subject', 'BlockSafe Test Email')
    body = data.get('body', '<p>This is a test email from BlockSafe.</p>')
    if not to:
        return jsonify({'error':'"to" field required'}), 400
    res = send_email(to, subject, body)
    print(f"DEBUG /admin/email-test to={to} res={res}")
    return jsonify({'to': to, 'result': res})

# ─────────────────────────────────────────────────────────────────────────────
# AGENT ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/agent')
@login_required
def agent_dashboard():
    if current_user.role != 'agent': return redirect(url_for('login'))
    return render_template('agent.html',
        username=current_user.username, bank_name=current_user.bank_name or 'Your Bank')

import uuid

@app.route('/agent/submit', methods=['POST'])
@login_required
def agent_submit():
    if current_user.role != 'agent': return jsonify({'error':'Agent access only'}), 403
    try:
        data   = request.get_json()
        result = run_prediction(data)
        # Auto-generate transaction ID
        tx_id = f"TXN-{datetime.utcnow().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8].upper()}"
        atx = AgentTransaction(
            agent_id=current_user.id,
            transaction_id=tx_id,
            bank_name=current_user.bank_name or data.get('bank_name',''),
            amount=result['amount'], age=result['age'], income=result['income'],
            debt=result['debt'], credit_score=result['credit_score'],
            hour=result['hour'], day=result['day'], month=result['month'], year=result['year'],
            verdict=result['verdict'], probability=result['probability'], model_used=result['model_used']
        )
        db.session.add(atx)
        if result['verdict'] == 'FRAUD':
            notif = Notification(user_id=current_user.id,
                message=f"🚨 FRAUD DETECTED — Transaction #{tx_id} | "
                        f"Amount: ${result['amount']:,.2f} | Probability: {result['probability']}% | "
                        f"Bank: {current_user.bank_name}")
            db.session.add(notif)
        db.session.commit()
        result['transaction_id'] = tx_id
        return jsonify(result)
    except Exception as e: return jsonify({'error':str(e)}), 400

@app.route('/agent/history')
@login_required
def agent_history():
    if current_user.role != 'agent': return jsonify({'error':'Forbidden'}), 403
    txs = AgentTransaction.query.filter_by(agent_id=current_user.id).order_by(AgentTransaction.id.desc()).all()
    return jsonify({'history':[{
        'id':t.id,'transaction_id':t.transaction_id,'bank_name':t.bank_name,
        'amount':t.amount,'verdict':t.verdict,'probability':t.probability,'model_used':t.model_used,
        'created_at':t.created_at.strftime('%Y-%m-%d %H:%M') if t.created_at else ''
    } for t in txs]})

@app.route('/agent/notifications')
@login_required
def agent_notifications():
    if current_user.role != 'agent': return jsonify({'error':'Forbidden'}), 403
    notifs = Notification.query.filter_by(user_id=current_user.id).order_by(Notification.id.desc()).limit(50).all()
    unread = Notification.query.filter_by(user_id=current_user.id, is_read=False).count()
    return jsonify({'notifications':[{
        'id':n.id,'message':n.message,'is_read':n.is_read,
        'created_at':n.created_at.strftime('%Y-%m-%d %H:%M') if n.created_at else ''
    } for n in notifs],'unread':unread})

@app.route('/agent/notifications/read', methods=['POST'])
@login_required
def mark_notifications_read():
    if current_user.role != 'agent': return jsonify({'error':'Forbidden'}), 403
    Notification.query.filter_by(user_id=current_user.id, is_read=False).update({'is_read':True})
    db.session.commit()
    return jsonify({'success':True})

if __name__ == '__main__':
    app.run(debug=True)