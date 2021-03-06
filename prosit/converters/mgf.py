from ..constants import MAX_ION, ION_TYPES, MAX_FRAG_CHARGE, NLOSSES
from .. import utils
from math import inf
import time
import pyteomics.mass
import numpy as np

aa_comp = dict(pyteomics.mass.std_aa_comp)
aa_comp["o"] = pyteomics.mass.Composition({"O": 1})
class Converter:
    def __init__(self, data, out_path):
        self.data = data
        self.out_path = out_path
        self.ion_list = self.create_ion_list()

    def create_ion_list(self):
        """Creates an ordered list of ion names (consisting of Fragment Number, Type, Charge and Neutral Loss type).
        The list is created dynamically because its shape depends on the number of allowed Neutral Loss types)
        """
        number_of_nlosses = self.data['intensities_pred'][0].size // 174
        nloss = NLOSSES[:number_of_nlosses]
        shape = [MAX_ION, len(ION_TYPES),  len(nloss), MAX_FRAG_CHARGE]
        FragmentNumber = np.zeros(shape, dtype=int)
        FragmentType = np.zeros(shape, dtype="object")
        FragmentCharge = np.zeros(shape, dtype=int)
        FragmentNeutralLoss = np.zeros(shape, dtype="object")
        for z in range(MAX_FRAG_CHARGE):
            for nli, nl in enumerate(nloss):
                for j in range(MAX_ION):
                    for tyi, ty in enumerate(ION_TYPES):
                        FragmentNumber[j, tyi, nli, z] = j + 1
                        FragmentType[j, tyi, nli, z] = ty
                        FragmentCharge[j, tyi, nli, z] = z + 1
                        FragmentNeutralLoss[j, tyi, nli, z] = nl

        FragmentNumber = FragmentNumber.flatten()
        FragmentType = FragmentType.flatten()
        FragmentCharge = FragmentCharge.flatten()
        FragmentNeutralLoss = FragmentNeutralLoss.flatten()
        return(np.asarray(["{}{}{}+{}".format(fragtype, fragnumber, fragnl if fragnl == '' else '-{}'.format(fragnl), fragcharge) 
            for fragtype, fragnumber, fragnl, fragcharge in zip(FragmentType, FragmentNumber, FragmentNeutralLoss, FragmentCharge)]))

    def convert(self, as_speclib=False):
        """If as_speclib is set to True, the Title of each spectrum will only contain the scan number 
        """
        spectra = []
        min_mz, max_mz = inf, -inf 
        start = time.time()
        for i in range(len(self.data["iRT"])):
            spectrum = Spectrum(i,
                self.data["sequence_integer"][i], 
                self.data["masses_pred"][i], 
                self.data["intensities_pred"][i], 
                self.data["collision_energy_aligned_normed"][i], 
                self.data["iRT"][i],
                self.data["precursor_charge_onehot"][i],
                self.ion_list)
            spectra.append(spectrum)
            mzlist = spectrum.masses_pred[spectrum.masses_pred!=-1]
            if mzlist.size!=0:
                min_mz = min(min_mz, mzlist.min())
                max_mz = max(max_mz, mzlist.max())
        print("Spectrum list generated: {:.3f}".format(time.time() - start))
        start = time.time()
        with open(self.out_path, "w") as outfile:
            outfile.write("MIN_MZ={:.6f}\nMAX_MZ={:.6f}\n".format(min_mz, max_mz))
            for spectrum in spectra:
                outfile.write(spectrum.to_mgf(as_speclib) + "\n")
        print("MGF file written: {:.3f}".format(time.time() - start))

class Spectrum(object):
    def __init__(self, index, sequence_integer, masses_pred, intensities_pred, collision_energy_aligned_normed, iRT, precursor_charge_onehot, ion_list):
        mod_sequence = utils.get_sequence(sequence_integer)
        self.index = index
        self.precursor_charge = int(precursor_charge_onehot.argmax() + 1)
        self.precursor_mz = pyteomics.mass.calculate_mass(
            sequence=mod_sequence.replace("M(ox)", "oM"), charge=self.precursor_charge, aa_comp=aa_comp)
        self.sequence, self.modifications = find_modifications(mod_sequence)
        self.ce = collision_energy_aligned_normed[0] #This is not yet printed anywhere, maybe someday
        self.iRT = iRT[0]
        ion_mask = np.round(intensities_pred,7) > 0
        self.masses_pred = masses_pred[ion_mask]
        self.intensities_pred = intensities_pred[ion_mask]
        self.ion_list = ion_list[ion_mask]
    
    def to_mgf(self, as_speclib):
        peak_list = ["{:.6f} {:.6f} {}".format(mz, rel_int, ion_string)
        for mz, rel_int, ion_string in zip(
            self.masses_pred,
            self.intensities_pred,
            self.ion_list)]
        res = "BEGIN IONS\nTITLE={}\nPEPMASS={:.6f}\nCHARGE={}\nIRT={:.6f}\n{}\nEND IONS\n".format(
            self.index if as_speclib else "{}|{}|{}".format(self.sequence,self.modifications,self.precursor_charge),
            calc_mass(self.precursor_mz, self.precursor_charge),
            self.precursor_charge,
            self.iRT,
            "\n".join(peak_list))
        return res
        
def find_modifications(peptide):
    res=""
    pos = peptide.find("M(")
    while pos != -1:
        res += "{},Oxidation[M];".format(pos+1)
        peptide = peptide[:peptide.find("(")] + peptide[peptide.find(")")+1:]
        pos = peptide.find("M(")
    return peptide, res

HYDROGEN_MASS = 1.00784
def calc_mass(mz, charge):
    return mz*charge - (charge*HYDROGEN_MASS)
