from flask import Flask, render_template, request
from utils import load_pickle, get_prediction, load_json

app = Flask(__name__)

classification_pipelines: dict[str] = load_pickle('notebooks/classification/files/pipeline_models.pkl')
classification_encoder = load_pickle('notebooks/classification/files/encoder.pkl')
classification_json_data = load_json('notebooks/classification/files/column_data.json')


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
    return render_template('clustering.html')


@app.route('/regression', methods=['GET', 'POST'])
def regression():
    return render_template('regression.html')


if __name__ == '__main__':
    app.run(debug=True)
