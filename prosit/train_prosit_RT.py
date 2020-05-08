import os
import h5py
import numpy as np
import pandas as pd
import yaml
from math import ceil
from . import model as model_lib
from . import training, tensorize, io_local, constants

#Turn off warnings:
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

#Load both training files at once and concatenate (for iRT, the fragmentation method does not matter so we can use both)
training_data_paths =  ["/root/training/IM_HCD_25_27_train_1_preprocessed.csv", "/root/training/IM_CID_35_train_1_preprocessed.csv"]
training_dfs = [pd.read_csv(path) for path in training_data_paths]
#DEBUG: Only use first 100 peptides
training_dfs = [df.iloc[:100] for df in training_dfs]
training_df = pd.concat(training_dfs)

###Dump all Peptides containing selenocystein
training_df = training_df.loc[~training_df.modified_sequence.str.contains("U")]
print("CSV Loaded, shape is {}.".format(training_df.shape))

###Prepare df for training
training_tensorized = tensorize.csv(training_df[['modified_sequence', 'collision_energy', 'precursor_charge']], nlosses=3)
print("CSV Tensorized.")
iRT_raw_mean = training_df.uRT.mean()
iRT_raw_var = training_df.uRT.var()
training_tensorized['prediction'] = np.reshape(
	np.asarray((training_df.uRT - iRT_raw_mean) / np.sqrt(iRT_raw_var)),(-1,1))

###Write and reload training data in hdf5 format
hdf5_path = "/root/training/training_data.hdf5"
io_local.to_hdf5(training_tensorized,hdf5_path)
print("Training Data Written to HDF5 File.")
#Load the hdf5 again
training_loaded = io_local.from_hdf5(hdf5_path)
print("Training Data Reloaded from HDF5 File.\nCommencing Training of iRT Model (with{} weights).".format("" if is_trained else "out"))

#The whole thing is run twice: Once with an untrained model and once with the pretrain model (to see if transfer learning is useful here)
for is_trained in [True, False]:
	###Load  Retention Time Model and prepare its training data 
	iRT_model_dir = "/root/training/IMA_uRT/"
	iRT_model, iRT_config = model_lib.load(iRT_model_dir, trained=is_trained)
	iRT_callbacks = training.get_callbacks(iRT_model_dir)
	print("iRT Model Loaded (with{} weights).".format("" if is_trained else "out"))

	#Write mean and var of iRTs to config file, they will be used at prediction time for scaling the results properly
	iRT_config['iRT_rescaling_mean'] = float(iRT_raw_mean)
	iRT_config['iRT_rescaling_var'] = float(iRT_raw_var)
	with open(iRT_model_dir + "config_new.yml", "w") as config_outfile:
	    yaml.dump(iRT_config, config_outfile)
	training.IMA_compile_model(iRT_model, iRT_config)

	###Train the model
	iRT_history = training.IMA_train(training_loaded, iRT_model, iRT_config, iRT_callbacks)
	iRT_epochs = len(iRT_history.history['val_loss'])
	iRT_val_loss = iRT_history.history['val_loss'][-1]
	iRT_weights_filename = "{}/{}weight_{:02d}_{:.5f}.hdf5".format(iRT_model_dir, "transfer_trained_" if is_trained else "", iRT_epochs, iRT_val_loss)
	iRT_model.save_weights(iRT_weights_filename)
	print("Training of iRT Model Complete.")
print("Done! You may now use these two models for your predictions.")
