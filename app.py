from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import numpy as np
import pickle

app = Flask(__name__)

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
ft_scaler = ft_bundle['scaler']
ft_model.eval()

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

import pandas as pd

NUM_COLS_9_FT  = ['amount', 'age', 'income', 'debt', 'credit_score', 'year', 'month', 'day', 'hour']
NUM_COLS_9_MLP = ['amount', 'age', 'income', 'debt', 'credit_score', 'hour', 'day', 'month', 'year']
NUM_COLS_14    = ['amount', 'age', 'income', 'debt', 'credit_score', 'hour', 'day', 'month', 'year',
                  'debt_to_income', 'amount_to_income', 'amount_to_debt', 'credit_x_amount', 'age_x_debt']
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

        # 9 features — FTTransformer order
        features_9_ft = pd.DataFrame([[
            amount, age, income, debt, credit_score,
            year, month, day, hour
        ]], columns=NUM_COLS_9_FT)

        # 9 features — MLP order
        features_9_mlp = pd.DataFrame([[
            amount, age, income, debt, credit_score,
            hour, day, month, year
        ]], columns=NUM_COLS_9_MLP)

        # 14 extracted features
        features_14 = pd.DataFrame([[
            amount, age, income, debt, credit_score,
            hour, day, month, year,
            debt_to_income, amount_to_income, amount_to_debt,
            credit_x_amount, age_x_debt
        ]], columns=NUM_COLS_14)

        if model_type == 'ft_original':
            scaled = ft_scaler.transform(features_9_ft)
            x_num  = torch.tensor(scaled, dtype=torch.float32)
            x_cat  = torch.tensor([[0, 0]], dtype=torch.long)
            with torch.no_grad():
                output = torch.sigmoid(ft_model(x_num, x_cat))
            label = 'FTTransformer — Original'

        elif model_type == 'ft_extracted':
            scaled = ft_ext_scaler.transform(features_14)
            x_num  = torch.tensor(scaled, dtype=torch.float32)
            x_cat  = torch.tensor([[0, 0]], dtype=torch.long)
            with torch.no_grad():
                output = torch.sigmoid(ft_ext_model(x_num, x_cat))
            label = 'FTTransformer — Extracted'

        elif model_type == 'mlp_original':
            scaled = mlp_scaler.transform(features_9_mlp)
            x      = torch.tensor(scaled, dtype=torch.float32)
            with torch.no_grad():
                output = torch.sigmoid(mlp_model(x))
            label = 'MLP — Original'

        else:  # mlp_extracted
            scaled = mlp_ext_scaler.transform(features_14)
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
