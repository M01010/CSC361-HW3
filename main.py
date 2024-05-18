from flask import Flask, render_template, request
from utils import load_pickle, get_prediction, load_json

app = Flask(__name__)

classification_pipelines = load_pickle('notebooks/classification/files/pipeline_models.pkl')
classification_encoder = load_pickle('notebooks/classification/files/encoder.pkl')
classification_json_data = load_json('notebooks/classification/files/column_data.json')

regression_pipelines = load_pickle('notebooks/regression/files/pipeline_models.pkl')
regression_json_data = load_json('notebooks/regression/files/column_data.json')

clustering_pipelines = load_pickle('notebooks/clustering/files/pipeline_models.pkl')
clustering_json_data = load_json('notebooks/clustering/files/column_data.json')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/classification', methods=['GET', 'POST'])
def classification():
    if request.method == 'GET':
        return render_template('classification.html',
                               models=classification_pipelines.keys(),
                               column_data=classification_json_data)
    else:
        model, y_label = get_prediction(request.form, classification_pipelines, classification_encoder)
        return render_template('classification.html',
                               model=model,
                               prediction=y_label)


@app.route('/clustering', methods=['GET', 'POST'])
def clustering():
    if request.method == 'GET':
        return render_template('clustering.html',
                               models=clustering_pipelines.keys(),
                               column_data=clustering_json_data)
    else:
        model, clustering_results = get_prediction(request.form, clustering_pipelines)
        print(clustering_results)
        return render_template('clustering.html',
                               model=model,
                               prediction=clustering_results)


@app.route('/regression', methods=['GET', 'POST'])
def regression():
    if request.method == 'GET':
        return render_template('regression.html',
                               models=regression_pipelines.keys(),
                               column_data=regression_json_data)
    else:
        model, y = get_prediction(request.form, regression_pipelines)
        return render_template('regression.html',
                               model=model,
                               prediction=y)


if __name__ == '__main__':
    app.run(debug=True)
