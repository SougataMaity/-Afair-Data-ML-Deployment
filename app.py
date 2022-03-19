from flask import Flask, render_template, request
import model
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return render_template ('home.html')
@app.route('/pred', methods = ['POST',"GET"])
def predict():
    #mar_rate,age,nym,children,religious,education,occupation,occupationHusb = int()
    if request.method == "POST":
        mar_rate = request.form.get("marrageRate")
        age = request.form.get("age")
        nym = request.form.get("no-year-marrage")
        children = request.form.get("no-of-children")
        religious = request.form.get("religious")
        education = request.form.get("education")
        occupation = request.form.get("occupation")
        occupationHusb = request.form.get("occupation-husb")

        input = np.array([mar_rate, age, nym,children,religious,education,occupation,occupationHusb])
        value = input.astype(np.float_)
        pred = model.lr_model.predict([value])[0]

        result = ''
        if pred == 1:
            result = 'Affair'
        else:
            result = 'No Affair'

    return render_template('predict.html', pred = '{}'.format(result))


if __name__ == '__main__':
    app.run(debug=True)