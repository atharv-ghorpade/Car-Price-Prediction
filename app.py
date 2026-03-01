from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

model = pickle.load(open('model/LinearRegressionModel.pkl','rb'))
car = pd.read_csv('data/cleaned car.csv')


@app.route('/')
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    fuel_type = car['fuel_type'].unique()

    return render_template(
        'index.html',
        companies=companies,
        car_models=car_models,
        fuel_type=fuel_type
    )


@app.route('/predict', methods=['POST'])
def predict():

    name = request.form.get('name')
    company = request.form.get('company')
    year = int(request.form.get('year'))
    kms_driven = int(request.form.get('kms'))
    fuel_type = request.form.get('fuel')

    input_df = pd.DataFrame(
        [[name, company, year, kms_driven, fuel_type]],
        columns=['name','company','year','kms_driven','fuel_type']
    )

    prediction = model.predict(input_df)

    return str(round(prediction[0],2))


if __name__ == "__main__":
    app.run(debug=True)