from flask import Flask, request, jsonify, render_template, send_file
import torch
import torch.nn as nn
import numpy as np
import pickle
import asyncio
import edge_tts
import tempfile

app = Flask(__name__)

VOICE = "en-US-JennyNeural"  # hardcoded US neural voice

@app.route('/speak')
def speak_route():
    text = request.args.get('text', '')
    if not text:
        return jsonify({'error': 'no text'}), 400
    tmp = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
    tmp.close()
    async def _generate():
        tts = edge_tts.Communicate(text=text, voice=VOICE)
        await tts.save(tmp.name)
    asyncio.run(_generate())
    return send_file(tmp.name, mimetype='audio/mpeg')

# ── Model Definitions ─────────────────────────────────────────────────────────
class FTTransformer(nn.Module):
    def __init__(self, num_numeric, num_locations, num_devices, embed_dim=8):
        super(FTTransformer, self).__init__()
        self.location_emb = nn.Embedding(num_locations, embed_dim)
        self.device_emb   = nn.Embedding(num_devices,   embed_dim)
        input_dim = num_numeric + embed_dim + embed_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x_num, x_cat):
        loc = self.location_emb(x_cat[:, 0].long())
        dev = self.device_emb(x_cat[:, 1].long())
        x   = torch.cat((x_num, loc, dev), dim=1)
        return self.model(x)

class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.4),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.model(x)

# ── Load All 4 Models ─────────────────────────────────────────────────────────
device = torch.device('cpu')

with open('fttransformer.pkl', 'rb') as f:
    ft_bundle = pickle.load(f)
ft_model  = ft_bundle['model'].to(device)
ft_model.eval()

# The scaler in fttransformer.pkl was fitted on already-scaled data (Cell 91
# scaled df in-place before saving). Rebuild a correct scaler from known ranges.
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
_ft_ranges = {
    'amount':       (1.0,   4999.0),
    'age':          (18.0,  70.0),
    'income':       (20000.0, 149999.0),
    'debt':         (0.0,   49999.0),
    'credit_score': (300.0, 850.0),
    'hour':         (0.0,   23.0),
    'day':          (1.0,   31.0),
    'month':        (1.0,   12.0),
    'year':         (2020.0, 2024.0),
}
_ft_cols = ['amount','age','income','debt','credit_score','hour','day','month','year']
ft_scaler = MinMaxScaler()
ft_scaler.fit(pd.DataFrame(
    [[_ft_ranges[c][0] for c in _ft_cols],
     [_ft_ranges[c][1] for c in _ft_cols]],
    columns=_ft_cols
))

with open('fttransformer_extracted.pkl', 'rb') as f:
    ft_ext_bundle = pickle.load(f)
ft_ext_model  = ft_ext_bundle['model'].to(device)
ft_ext_scaler = ft_ext_bundle['scaler']
ft_ext_model.eval()

with open('mlp.pkl', 'rb') as f:
    mlp_bundle = pickle.load(f)
mlp_model  = mlp_bundle['model'].to(device)
mlp_scaler = mlp_bundle['scaler']
mlp_model.eval()

with open('mlp_extracted.pkl', 'rb') as f:
    mlp_ext_bundle = pickle.load(f)
mlp_ext_model  = mlp_ext_bundle['model'].to(device)
mlp_ext_scaler = mlp_ext_bundle['scaler']
mlp_ext_model.eval()



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data       = request.get_json()
        model_type = data.get('model_type', 'ft_original')

        amount       = float(data['amount'])
        age          = float(data['age'])
        income       = float(data['income'])
        debt         = float(data['debt'])
        credit_score = float(data['credit_score'])
        hour         = float(data['hour'])
        day          = float(data['day'])
        month        = float(data['month'])
        year         = float(data['year'])

        # Extracted features
        debt_to_income   = debt   / (income + 1e-6)
        amount_to_income = amount / (income + 1e-6)
        amount_to_debt   = amount / (debt   + 1e-6)
        credit_x_amount  = credit_score * amount
        age_x_debt       = age * debt

        # Use numpy arrays to bypass sklearn feature-name validation entirely
        arr_9  = np.array([[amount, age, income, debt, credit_score,
                            hour, day, month, year]])
        arr_14 = np.array([[amount, age, income, debt, credit_score,
                            hour, day, month, year,
                            debt_to_income, amount_to_income, amount_to_debt,
                            credit_x_amount, age_x_debt]])

        if model_type == 'ft_original':
            scaled = ft_scaler.transform(arr_9)
            x_num  = torch.tensor(scaled, dtype=torch.float32)
            x_cat  = torch.tensor([[0, 0]], dtype=torch.long)
            with torch.no_grad():
                output = torch.sigmoid(ft_model(x_num, x_cat))
            label = 'FTTransformer — Original'

        elif model_type == 'ft_extracted':
            scaled = ft_ext_scaler.transform(arr_14)
            x_num  = torch.tensor(scaled, dtype=torch.float32)
            x_cat  = torch.tensor([[0, 0]], dtype=torch.long)
            with torch.no_grad():
                output = torch.sigmoid(ft_ext_model(x_num, x_cat))
            label = 'FTTransformer — Extracted'

        elif model_type == 'mlp_original':
            scaled = mlp_scaler.transform(arr_9)
            x      = torch.tensor(scaled, dtype=torch.float32)
            with torch.no_grad():
                output = torch.sigmoid(mlp_model(x))
            label = 'MLP — Original'

        else:  # mlp_extracted
            scaled = mlp_ext_scaler.transform(arr_14)
            x      = torch.tensor(scaled, dtype=torch.float32)
            with torch.no_grad():
                output = torch.sigmoid(mlp_ext_model(x))
            label = 'MLP — Extracted'

        probability = output.item()
        verdict     = 'FRAUD' if probability >= 0.5 else 'LEGITIMATE'
        confidence  = probability if verdict == 'FRAUD' else 1 - probability

        return jsonify({
            'verdict':     verdict,
            'probability': round(probability * 100, 2),
            'confidence':  round(confidence * 100, 2),
            'model_used':  label
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
