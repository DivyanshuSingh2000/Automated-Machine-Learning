from flask import Flask, render_template, request, jsonify
from flask_cors import cross_origin
import os

import classification
import regression
from readAndProcess import BackEndWork

app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
@cross_origin()
def homepage():
    return render_template("home.html")


@app.route("/review", methods=['GET', 'POST'])
@cross_origin()
def index():

    if request.method == 'POST':

        f = request.files['ufile']
        # check if any file is selected or not

        if f.filename == '':
            return jsonify("Please Upload a file .... You Haven't Selected a file")
        f.save(os.path.join("uploaded_Files", f.filename))

        f_type = request.form["ftype"]  # file type (csv, json, excel)
        p_type = request.form["ptype"]  # problem type
        target = str(request.form.get("target", False))

        obj = BackEndWork(f.filename, f_type)
        obj.reading()  # read the file
        df_num, df_obj = obj.separating_df()
        df = obj.preprocessing_df(df_num, df_obj)

        # to check problem type and create object according to the problem type
        if p_type == "classification":
            training_obj = classification.TrainAndEvaluate(df, target)
        else:
            training_obj = regression.TrainAndEvaluate(df, target)

        models = training_obj.model_tune()
        d = training_obj.best_model(models)
        keymax = sorted(d, key=d.get, reverse=True)[:3]

        return render_template('results.html', di=d, km=keymax)


if __name__ == "__main__":
    app.run(debug=True)
