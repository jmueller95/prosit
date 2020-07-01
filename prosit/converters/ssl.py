from .. import utils
import pandas as pd
import numpy as np

class Converter:
	def __init__(self, data, out_path):
		self.data = data
		self.out_path = out_path

	def convert(self):
		res = pd.DataFrame({
			"file":"{}.mgf".format(".".join(self.out_path.split("/")[-1].split(".")[:-1])),
			"scan": range(len(self.data)),
			"charge": self.data['precursor_charge'],
			"sequence": self.data['modified_sequence'].apply(lambda seq: seq.replace("M(ox)", "M[+16.0]"))
			})
		res[['file', 'scan', 'charge', 'sequence']].to_csv(self.out_path, sep="\t", header=True, index=False)
