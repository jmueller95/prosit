import os
import numpy as np
import pandas as pd
import yaml
from . import model as model_lib
from . import training, tensorize, io_local

def main():
	#Turn off warnings:
	os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

	###Load training data - Put the path to your own data here
	training_data_path = "/root/training/training_preprocessed.csv"
	training_df = pd.read_csv(training_data_path)

	###Dump all Peptides containing selenocystein
	training_df = training_df.loc[~training_df.modified_sequence.str.contains("U")]

	print("CSV Loaded, shape is {}.".format(training_df.shape))
	###Load Untrained Retention Time Model and prepare its training data
	iRT_model_dir = "/root/training/iRT/"
	iRT_model, iRT_config = model_lib.load(iRT_model_dir, trained=False)
	iRT_callbacks = training.get_callbacks(iRT_model_dir)
	iRT_raw_mean = training_df.uRT.mean()
	iRT_raw_var = training_df.uRT.var()
	iRT_config['iRT_rescaling_mean'] = float(iRT_raw_mean)
	iRT_config['iRT_rescaling_var'] = float(iRT_raw_var)
	with open(iRT_model_dir + "config_new.yml", "w") as config_outfile:
		yaml.dump(iRT_config, config_outfile)
	###Load Untrained Fragmentation Model and prepare its training data
	msms_model_dir = "/root/training/msms/"
	msms_model, msms_config = model_lib.load(msms_model_dir, trained=False)
	msms_callbacks = training.get_callbacks(msms_model_dir)
	#The intensity lists are already in proper order, but might have some missing values and need to be padded to the correct length
	#(Only a peptide of the maximal length 29 will have 522 values, but all lists need to be of this length)
	intensities_length = 522
	print("iRT and Fragmentation Intensity Models Loaded.")


	#Compile the models once, and then call fit separately - useful if you lack memory or space and have to partition your training data
	training.compile_model(iRT_model, iRT_config)
	training.compile_model(msms_model, msms_config)

	training_tensorized = tensorize.csv(training_df[['modified_sequence', 'collision_energy', 'precursor_charge']], nlosses=3)

	print("CSV Tensorized.")
	training_tensorized['prediction'] = np.reshape(
		np.asarray((training_df.uRT - iRT_raw_mean) / np.sqrt(iRT_raw_var)),
		(-1,1))
	training_df.relative_intensities = training_df.relative_intensities.apply(eval)
	training_df.relative_intensities = training_df.relative_intensities.apply(
		lambda ls: np.nan_to_num(np.pad(ls, pad_width=(0,intensities_length-len(ls)),constant_values=-1, mode="constant"),-1))
	training_tensorized['intensities_raw'] = np.stack(training_df.relative_intensities)


	###Write and reload training data in hdf5 format
	hdf5_path = "/root/training/training_data.hdf5"
	io_local.to_hdf5(training_tensorized,hdf5_path)
	print("Training Data Written to HDF5 File.")
	#Load the hdf5 again
	training_loaded = io_local.from_hdf5(hdf5_path)
	print("Training Data Reloaded from HDF5 File.\nCommencing Training of iRT Model...")


	###Train both models
	iRT_history = training.train_model(training_loaded, iRT_model, iRT_config, iRT_callbacks)
	iRT_epochs = len(iRT_history.history['val_loss'])
	iRT_val_loss = iRT_history.history['val_loss'][-1]
	iRT_weights_filename = "{}/weight_{:02d}_{:.5f}.hdf5".format(iRT_model_dir, iRT_epochs, iRT_val_loss)
	iRT_model.save_weights(iRT_weights_filename)
	print("Training of iRT Model Complete.\nCommencing Training of Fragmentation Intensity Model...")
	msms_history = training.train_model(training_loaded, msms_model, msms_config, msms_callbacks)
	#Save the weights to a file named by the val_loss and the epochs
	msms_epochs = len(msms_history.history['val_loss'])
	msms_val_loss = msms_history.history['val_loss'][-1]
	msms_weights_filename = "{}/weight_{:02d}_{:.5f}.hdf5".format(msms_model_dir, msms_epochs, msms_val_loss)
	msms_model.save_weights(msms_weights_filename)
	print("Training of Fragmentation Intensity Model Complete.")
	print("Done! You may now use these models for your predictions.")


if __name__ == '__main__':
	main()
