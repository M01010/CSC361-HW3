import numpy as np
from flask import Flask, render_template, request
from utils import load_model, get_models

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.get('/classification')
def classification_get():
    models = get_models('notebooks/classification/models')
    print(models)
    return render_template('classification.html', models=models)


@app.post('/classification')
def classification_post():
    model_name = request.form['model']
    x_dict = {}
    for key, value in request.form.items():
        if key != 'model':
            x_dict[key] = value

    x = np.array([list(float(i) for i in x_dict.values())])
    model = load_model(f"notebooks/classification/models/{model_name}")
    prediction = model.predict(x)
    print(model, x, prediction)
    return render_template('classification.html', model=model.__class__.__name__, prediction=prediction)


@app.get('/clustering')
def clustering_get():
    return render_template('clustering.html')


@app.post('/clustering')
def clustering_post():
    return render_template('clustering.html')


@app.get('/regression')
def regression_get():
    return render_template('regression.html')


@app.post('/regression')
def regression_post():
    return render_template('regression.html')


if __name__ == '__main__':
    app.run(debug=True)
