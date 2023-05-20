import os
from io import StringIO
import csv

from flask import (
    Flask,
    request,
    render_template,
    make_response
)
import pandas as pd
import numpy as np

from src.utils.predict import predict

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_csv', methods=["POST"])
def transform_view():
    f = request.files['data_file']
    if not f:
        return "No file"

    stream = StringIO(f.stream.read().decode("UTF8"), newline=None)
    csv_input = csv.reader(stream)
    data = []
    for row in csv_input:
        data.append(row)
    data = pd.DataFrame(data[1:], columns=data[0])

    data = predict(data)

    response = make_response(data.to_csv())
    response.headers["Content-Disposition"] = "attachment; filename=predictions.csv"
    return response

@app.route('/predict_json', methods=["POST"])
def results():
    json_input = request.get_json(force=True)
    print(json_input)
    data = pd.json_normalize(json_input)
    data = predict(data)
    json_out = data.to_json(orient="records", force_ascii=False, indent=4)
    return json_out

if __name__ == "__main__":
    app.run(port=5000, debug=True)