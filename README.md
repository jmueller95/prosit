# Prosit
Forked from https://github.com/kusterlab/prosit. 
This project extends Prosit from the Kusterlab@TUM by several small features. 
Prosit is a deep neural network to predict iRT values and MS2 spectra for given peptide sequences.
For information on features, usage, and requirements of Prosit in general, please refer to the readme in the original repository.

## Neutral Loss Predictions
In addition to _b_ and _y_ ions, the neural network architecture provided here is able to predict water (_-H2O_) and ammonia
 (_-NH3_) neutral loss peaks in MS/MS spectra. Please note that before using this feature, you must train a fragmentation model on data that includes neutral loss peaks.
While it is possible to employ the model provided by the Kusterlab (https://figshare.com/projects/Prosit/35582) to make use of the other features of this project,
but since this model was not trained on any neutral loss data, no neutral loss peaks can be predicted by it.

## MGF Output
In addition to the `generic`, `msms`, and `msp` outputs, this software can also output `mgf` files by using the command   
```curl -F "peptides=@<your_peptidelist.csv>" http://<your_server>/predict/mgf```  
(please see `examples/peptidelist.mgf` for an example). When predicting spectra with neutral losses,
it is advised to use this format because it includes the predicted retention times (which `msp` and `msms` do not), 
and is rather compact in comparison to the `generic` format, which does not scale well if neutral loss peaks are included
and you want to predict a large number of spectra. 

## Spectral Library Output 
If you want to create spectral libraries for DIA analysis with a software such as Skyline (https://skyline.ms/),
you need an SSL file in addition to the MGF file. The new route   
```curl -F "peptides=@<your_peptidelist.csv>" http://<your_server>/predict/speclib```  
returns a zip file containing such an SSL file and an MGF where the `TITLE`s map to the scan numbers in the SSL file 
(which is the input format Skyline expects). See `examples/peptidelist_speclib.[ssl|mgf]` for an example.
## RT Prediction Only
Sometimes you are only interested in the predicted retention times of your peptides. Prosit performs this prediction much faster than the prediction of fragmentation patterns,
but until now, this property could not be exploited because the interface only allowed combined predictions. 
This project implements another route which can be called by   
```curl -F "peptides=@<your_peptidelist.csv>" http://<your_server>/predict/rt```  
and only predicts the retention times of the input peptides in considerably less time compared to the full routes. Since Prosit's retention time model
only relies on the peptide sequence, the input csv file need only consist of one column `modified_sequence`. See `examples/peptideslist_rt.tsv`
for an exemplary output.

## No Fixed CAM
In the original version of Prosit, each C in an input sequence is treated as Cysteine with carbamidomethylation,
since this is a fixed modification in MaxQuant. Since the analysis pipeline for which this project was developed did not use MaxQuant
and the peptides used in the underlying models did not contain any modified Cysteins, this modification was removed.  
Note that because of this, it is discouraged to use the model trained by the Kusterlab (https://figshare.com/projects/Prosit/35582)
on this project, as it cannot be guaranteed that carbamidomethylated Cysteins behave in the same way as unmodified Cysteins
in an LC/MSMS experiment. 