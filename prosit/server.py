import time
import os
import tempfile
import warnings
import flask
from flask import after_this_request
import pandas as pd
import tensorflow as tf
import argparse
import zipfile
import io

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


def predict(df, mode, fragmentation_mode=None, nlosses=None):
    """ nlosses and fragmentation mode are only required if MSMS prediction should be included
        fragmentation_mode must be one of "CID", "HCD"
        mode: String, one of 'rt', 'msms', 'both'
    """

    #If msms prediction is included, convert and predict the input df. Else only convert the input for rt prediction 
    data = prediction.predict(tensorize.csv(df, nlosses), d_spectra[fragmentation_mode], nlosses) if mode != "rt" else tensorize.csv_only_seq(df)
    #Perform rt prediction if requested
    if mode != "msms":
        data = prediction.predict(data, d_irt)

    return data

@app.route("/predict/<output_format>", methods=["POST"])
@app.route("/predict/<output_format>/<fragmentation_mode>", methods=["POST"])
def run_prosit(output_format=None, fragmentation_mode=None):
    df = pd.read_csv(flask.request.files['peptides'])
    result = predict(df, 
        mode="rt" if output_format == "rt" else "both", #Isolated msms prediction is currently not possible, but there's no reason to do that right now.
        fragmentation_mode =  fragmentation_mode.upper() if fragmentation_mode else None,
        nlosses = 3) ###Neutral Losses currently are always predicted by default.
    if output_format not in ["speclib", "msms"]:
        tmp_f = tempfile.NamedTemporaryFile(delete=True)
        if output_format == "rt":
            c = converters.rtlist.Converter(result, tmp_f.name)
        elif output_format == "mgf":
            c = converters.mgf.Converter(result, tmp_f.name)
        elif output_format == "generic":
            c = converters.generic.Converter(result, tmp_f.name)
        elif output_format == "msp":
            print("Warning: msp output is not yet implemented!")
            c = converters.msp.Converter(result, tmp_f.name)
        c.convert()
    elif output_format == "msms":
        print("Warning: msms output is not yet implemented!")
        df_pred = converters.maxquant.convert_prediction(result)
        tmp_f = tempfile.NamedTemporaryFile(delete=True)
        converters.maxquant.write(df_pred, tmp_f.name)

    if output_format != "speclib":
        @after_this_request
        def cleanup(response):
            tmp_f.close()
            return response

        return flask.send_file(tmp_f.name)

    else:
        peptides_filename = ".".join(flask.request.files["peptides"].filename.split("/")[-1].split(".")[:-1])
        zipdata = io.BytesIO()
        with zipfile.ZipFile(zipdata, 'w', zipfile.ZIP_DEFLATED) as zipf:
            mgf_file = "{}.mgf".format(peptides_filename)
            c_mgf = converters.mgf.Converter(result, mgf_file)
            c_mgf.convert(as_speclib=True)
            zipf.write(mgf_file)
            # SSL only needs the input data, not the predictions
            ssl_file = "{}.ssl".format(peptides_filename)
            c_ssl = converters.ssl.Converter(df, ssl_file)
            c_ssl.convert()
            zipf.write(ssl_file)

        @after_this_request
        def cleanup(response):
            os.remove("{}.mgf".format(peptides_filename))
            os.remove("{}.ssl".format(peptides_filename))
            return response

        zipdata.seek(0)
        return flask.send_file(zipdata, mimetype='zip')

if __name__ == "__main__":
    ###################################
    # Have Keras allocate memory only when needed
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #This allows the user to customize the port the server uses
    parser = argparse.ArgumentParser()
    parser.add_argument("-p" "--port", action="store", dest="port")
    args = parser.parse_args()
    ###################################
    warnings.filterwarnings("ignore")
    global d_spectra
    global d_irt
    
    

    #Load CID MSMS Model
    d_spectra_cid = {}
    d_spectra_cid["graph"] = tf.Graph()
    with d_spectra_cid["graph"].as_default():
        d_spectra_cid["session"] = tf.Session(config=config)
        with d_spectra_cid["session"].as_default():
            d_spectra_cid["model"], d_spectra_cid["config"] = model.load(
                constants.MODEL_CONFIG_SPECTRA,
                constants.WEIGHTS_CID,
                trained=True
            )
            d_spectra_cid["model"].compile(optimizer="adam", loss="mse")

    #Load HCD MSMS Model
    d_spectra_hcd = {}
    d_spectra_hcd["graph"] = tf.Graph()
    with d_spectra_hcd["graph"].as_default():
        d_spectra_hcd["session"] = tf.Session(config=config)
        with d_spectra_hcd["session"].as_default():
            d_spectra_hcd["model"], d_spectra_hcd["config"] = model.load(
                constants.MODEL_CONFIG_SPECTRA,
                constants.WEIGHTS_HCD,
                trained=True
            )
            d_spectra_hcd["model"].compile(optimizer="adam", loss="mse")

    d_spectra = {"CID": d_spectra_cid, "HCD": d_spectra_hcd}

    #Load RT Model
    d_irt = {}
    d_irt["graph"] = tf.Graph()
    with d_irt["graph"].as_default():
        d_irt["session"] = tf.Session(config=config)
        with d_irt["session"].as_default():
            d_irt["model"], d_irt["config"] = model.load(constants.MODEL_CONFIG_RT,
                                                         constants.WEIGHTS_RT,
                                                         trained=True)
            d_irt["model"].compile(optimizer="adam", loss="mse")
    app.run(host="0.0.0.0", port=args.port)
