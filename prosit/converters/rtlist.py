from .. import utils
import pandas as pd
import numpy as np

class Converter:
    def __init__(self, data, out_path):
        self.data = data
        self.out_path = out_path

    def convert(self):
        res = pd.DataFrame({
            "modified_sequence" : [utils.get_sequence(seq_int) for seq_int in self.data["sequence_integer"]],
            "iRT" : np.hstack(self.data["iRT"])
            })
        res[['modified_sequence', 'iRT']].to_csv(self.out_path, sep="\t", index=False)
