from flask import Flask, render_template, request
from utils import load_model

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.get('/classification')
def classification_get():
    return render_template('classification.html')


@app.post('/classification')
def classification_post():
    print(request.form.to_dict())
    return render_template('classification.html')


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
