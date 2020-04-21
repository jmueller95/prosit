import time
import os
import tempfile
import warnings
import flask
from flask import after_this_request
import pandas as pd
import tensorflow as tf
import argparse

from . import model
from . import io_local
from . import constants
from . import tensorize
from . import prediction
from . import alignment
from . import converters

app = flask.Flask(__name__)


@app.route("/")
def hello():
    return "prosit!\n"


def predict(csv, nlosses):
    df = pd.read_csv(csv)
    start = time.time()
    data = tensorize.csv(df, nlosses)
    print("Tensorize Input DF: {:.3f}".format(time.time() - start))
    start = time.time()
    data = prediction.predict(data, d_spectra, nlosses)
    print("Predict MSMS: {:.3f}".format(time.time() - start))
    start = time.time()
    data = prediction.predict(data, d_irt, nlosses)
    print("Predict iRT: {:.3f}".format(time.time() - start))

    return data


@app.route("/predict/mgf", methods=["POST"])
def return_mgf():
    result = predict(flask.request.files["peptides"], nlosses=3)
    tmp_f = tempfile.NamedTemporaryFile(delete=True)
    c = converters.mgf.Converter(result, tmp_f.name)
    c.convert()

    @after_this_request
    def cleanup(response):
        tmp_f.close()
        return response

    return flask.send_file(tmp_f.name)


@app.route("/predict/generic", methods=["POST"])
def return_generic():
    result = predict(flask.request.files["peptides"], nlosses=3)
    tmp_f = tempfile.NamedTemporaryFile(delete=True)
    start = time.time()
    c = converters.generic.Converter(result, tmp_f.name)
    print("Create Generic Converter: {:.3f}".format(time.time() - start))
    c.convert()

    @after_this_request
    def cleanup(response):
        tmp_f.close()
        return response

    return flask.send_file(tmp_f.name)

@app.route("/predict/msp", methods=["POST"])
def return_msp():
    result = predict(flask.request.files["peptides"], nlosses=3)
    tmp_f = tempfile.NamedTemporaryFile(delete=True)
    c = converters.msp.Converter(result, tmp_f.name)
    c.convert()

    @after_this_request
    def cleanup(response):
        tmp_f.close()
        return response

    return flask.send_file(tmp_f.name)

@app.route("/predict/msms", methods=["POST"])
def return_msms():
    result = predict(flask.request.files["peptides"], nlosses=3)
    df_pred = converters.maxquant.convert_prediction(result)
    tmp_f = tempfile.NamedTemporaryFile(delete=True)
    converters.maxquant.write(df_pred, tmp_f.name)

    @after_this_request
    def cleanup(response):
        tmp_f.close()
        return response

    return flask.send_file(tmp_f.name)


if __name__ == "__main__":
    ###################################
    #Edit by JuMu to have Keras allocate memory only when needed
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #Edit by JuMu to customize the port the server uses
    parser = argparse.ArgumentParser()
    parser.add_argument("-p" "--port", action="store", dest="port")
    args = parser.parse_args()
    ###################################
    warnings.filterwarnings("ignore")
    global d_spectra
    global d_irt
    d_spectra = {}
    d_irt = {}

    d_spectra["graph"] = tf.Graph()
    with d_spectra["graph"].as_default():
        d_spectra["session"] = tf.Session(config=config)#Edit by JuMu
        with d_spectra["session"].as_default():
            d_spectra["model"], d_spectra["config"] = model.load(
                constants.MODEL_SPECTRA,
                trained=True
            )
            d_spectra["model"].compile(optimizer="adam", loss="mse")
    d_irt["graph"] = tf.Graph()
    with d_irt["graph"].as_default():
        d_irt["session"] = tf.Session(config=config)#Edit by JuMu
        with d_irt["session"].as_default():
            d_irt["model"], d_irt["config"] = model.load(constants.MODEL_IRT,
                    trained=True)
            d_irt["model"].compile(optimizer="adam", loss="mse")
    app.run(host="0.0.0.0", port=args.port)
