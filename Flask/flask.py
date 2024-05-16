from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)

# Folder where model files are stored
MODEL_FOLDER = 'models'


# Function to load a specific model
def load_model(model_name):
    model_path = os.path.join(MODEL_FOLDER, model_name)
    # Load the model from the file
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


# Home page
@app.route('/')
def index():
    # Render the index.html template
    return render_template('index.html')


# Classification page
@app.route('/classification')
def classification():
    # Render the classification.html template
    return render_template('classification.html')


# Model selection page
@app.route('/model_selection', methods=['GET', 'POST'])
def model_selection():
    if request.method == 'POST':
        # If the form is submitted, get the selected model name
        selected_model_name = request.form['model']
        # Load the selected model
        selected_model = load_model(selected_model_name)

        # Get user input data from the form
        data = request.form.getlist('data')

        # Process the user input data and make predictions using the selected model
        prediction = selected_model.predict(data)

        # Render the model_result.html template with the selected model name and prediction
        return render_template('model_result.html', selected_model_name=selected_model_name, prediction=prediction)
    else:
        # If it's a GET request, list all model files in the model folder and render the model_selection.html template
        model_files = os.listdir(MODEL_FOLDER)
        return render_template('model_selection.html', model_files=model_files)


# Clustering page
@app.route('/clustering')
def clustering():
    # Render the clustering.html template
    return render_template('clustering.html')


# Regression page
@app.route('/regression')
def regression():
    # Render the regression.html template
    return render_template('regression.html')


if __name__ == '__main__':
    # Run the Flask application
    app.run()
