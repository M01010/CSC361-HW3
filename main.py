from flask import Flask, render_template, request
from utils import load_pickle, get_prediction, load_json

app = Flask(__name__)
classification_pipelines: dict[str] = load_pickle('notebooks/classification/pipeline_models.pkl')
classification_encoder = load_pickle('notebooks/classification/encoder.pkl')
classification_json_data = load_json('notebooks/classification/column_data.json')


@app.route('/')
def index():
    return render_template('index.html')


@app.get('/classification')
def classification_get():
    return render_template('classification.html',
                           models=classification_pipelines.keys(),
                           column_data=classification_json_data
                           )


@app.post('/classification')
def classification_post():
    model, y_label = get_prediction(request.form, classification_pipelines, classification_encoder)
    return render_template('classification.html',
                           model=model,
                           prediction=y_label)


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
