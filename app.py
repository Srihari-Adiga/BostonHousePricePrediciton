import pickle
from flask import Flask,render_template,request,app,jsonify,url_for
import numpy as np
import pandas as pd

app = Flask(__name__)
#Loading the model
regmodel = pickle.load(open('regmodel.pkl','rb'))
scalar = pickle.load(open('Scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(-1,1))
    new_data = scalar.transform(np.array(list(data.values())).reshape(1,-1))
    result = regmodel.predict(new_data)
    print(result[0])
    return jsonify(result[0])

@app.route('/predict',methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    result = regmodel.predict(final_input)[0]
    return render_template('home.html',predicted_text = "The house price predicted is {}".format(result))


if __name__=='__main__':
    app.run(debug=True)
