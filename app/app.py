from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

MODEL_PATH = r"D:\Kidney-Disease-Prediction\models\model.pkl"
model = joblib.load(MODEL_PATH)

FEATURE_NAMES = [
    'age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar',
    'red_blood_cells', 'pus_cell', 'pus_cell_clumps', 'bacteria',
    'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium',
    'potassium', 'hemoglobin', 'packed_cell_volume', 'white_blood_cell_count',
    'red_blood_cell_count', 'hypertension', 'diabetes_mellitus',
    'coronary_artery_disease', 'appetite', 'pedal_edema', 'anemia'
]

FEATURE_INFO = {
    'age':                    {'label': 'Tuổi',                      'unit': 'năm',    'min': 0,   'max': 90,   'step': 1,    'type': 'number'},
    'blood_pressure':         {'label': 'Huyết áp',                  'unit': 'mm/Hg',  'min': 50,  'max': 180,  'step': 1,    'type': 'number'},
    'specific_gravity':       {'label': 'Tỷ trọng nước tiểu',        'unit': '',       'min': 1.005,'max': 1.025,'step': 0.001,'type': 'number'},
    'albumin':                {'label': 'Albumin',                    'unit': '',       'min': 0,   'max': 5,    'step': 1,    'type': 'select', 'options': [0,1,2,3,4,5]},
    'sugar':                  {'label': 'Đường',                      'unit': '',       'min': 0,   'max': 5,    'step': 1,    'type': 'select', 'options': [0,1,2,3,4,5]},
    'red_blood_cells':        {'label': 'Hồng cầu',                  'unit': '',       'min': 0,   'max': 1,    'step': 1,    'type': 'toggle', 'options': ['Bất thường (0)', 'Bình thường (1)']},
    'pus_cell':               {'label': 'Tế bào mủ',                 'unit': '',       'min': 0,   'max': 1,    'step': 1,    'type': 'toggle', 'options': ['Bất thường (0)', 'Bình thường (1)']},
    'pus_cell_clumps':        {'label': 'Cụm tế bào mủ',             'unit': '',       'min': 0,   'max': 1,    'step': 1,    'type': 'toggle', 'options': ['Không có (0)', 'Có (1)']},
    'bacteria':               {'label': 'Vi khuẩn',                  'unit': '',       'min': 0,   'max': 1,    'step': 1,    'type': 'toggle', 'options': ['Không có (0)', 'Có (1)']},
    'blood_glucose_random':   {'label': 'Đường huyết ngẫu nhiên',    'unit': 'mgs/dl', 'min': 70,  'max': 500,  'step': 1,    'type': 'number'},
    'blood_urea':             {'label': 'Ure máu',                   'unit': 'mgs/dl', 'min': 1,   'max': 400,  'step': 1,    'type': 'number'},
    'serum_creatinine':       {'label': 'Creatinine huyết thanh',    'unit': 'mgs/dl', 'min': 0.4, 'max': 76,   'step': 0.1,  'type': 'number'},
    'sodium':                 {'label': 'Natri',                     'unit': 'mEq/L',  'min': 100, 'max': 165,  'step': 1,    'type': 'number'},
    'potassium':              {'label': 'Kali',                      'unit': 'mEq/L',  'min': 2.5, 'max': 47,   'step': 0.1,  'type': 'number'},
    'hemoglobin':             {'label': 'Hemoglobin',                'unit': 'gms',    'min': 3,   'max': 17.8, 'step': 0.1,  'type': 'number'},
    'packed_cell_volume':     {'label': 'Thể tích hồng cầu',         'unit': '%',      'min': 9,   'max': 54,   'step': 1,    'type': 'number'},
    'white_blood_cell_count': {'label': 'Số lượng bạch cầu',          'unit': 'cells/cumm','min': 2200,'max':26400,'step': 100, 'type': 'number'},
    'red_blood_cell_count':   {'label': 'Số lượng hồng cầu',         'unit': 'millions/cmm','min':2.1,'max':8,   'step': 0.1,  'type': 'number'},
    'hypertension':           {'label': 'Tăng huyết áp',             'unit': '',       'min': 0,   'max': 1,    'step': 1,    'type': 'toggle', 'options': ['Không (0)', 'Có (1)']},
    'diabetes_mellitus':      {'label': 'Tiểu đường',                'unit': '',       'min': 0,   'max': 1,    'step': 1,    'type': 'toggle', 'options': ['Không (0)', 'Có (1)']},
    'coronary_artery_disease':{'label': 'Bệnh mạch vành',            'unit': '',       'min': 0,   'max': 1,    'step': 1,    'type': 'toggle', 'options': ['Không (0)', 'Có (1)']},
    'appetite':               {'label': 'Cảm giác thèm ăn',          'unit': '',       'min': 0,   'max': 1,    'step': 1,    'type': 'toggle', 'options': ['Kém (0)', 'Tốt (1)']},
    'pedal_edema':            {'label': 'Phù chân',                  'unit': '',       'min': 0,   'max': 1,    'step': 1,    'type': 'toggle', 'options': ['Không (0)', 'Có (1)']},
    'anemia':                 {'label': 'Thiếu máu',                 'unit': '',       'min': 0,   'max': 1,    'step': 1,    'type': 'toggle', 'options': ['Không (0)', 'Có (1)']},
}

SAMPLE_CKD = [0.543, 0.875, 0.714, 0.4, 0.0, 0.0, 0.0, 1.0, 0.0,
              0.217, 0.939, 1.0, 0.0, 0.281, 0.339, 0.286, 0.903, 0.042,
              1.0, 1.0, 0.0, 0.0, 0.0, 1.0]

SAMPLE_NORMAL = [0.3, 0.25, 0.857, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
                 0.08, 0.05, 0.02, 0.7, 0.1, 0.85, 0.75, 0.35, 0.65,
                 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]

@app.route('/')
def index():
    return render_template('index.html',
                           features=FEATURE_NAMES,
                           feature_info=FEATURE_INFO,
                           sample_ckd=SAMPLE_CKD,
                           sample_normal=SAMPLE_NORMAL)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        values = [float(data[f]) for f in FEATURE_NAMES]
        input_df = pd.DataFrame([values], columns=FEATURE_NAMES)

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'probability': round(float(probability), 4),
            'label': 'CKD' if prediction == 1 else 'Not CKD',
            'message': 'Bệnh nhân CÓ nguy cơ bệnh thận mãn tính' if prediction == 1
                       else 'Bệnh nhân KHÔNG có nguy cơ bệnh thận mãn tính'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)