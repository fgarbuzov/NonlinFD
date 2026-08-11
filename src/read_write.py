import numpy as np
import pandas as pd
import os
import warnings

from enum import Enum
from pathlib import Path

from .tensor import lammps_array_to_matrix, voigt_to_tensor, sym_dyad

class Filenames(Enum):
    THERMO    = "thermo.dat"
    PRESS     = "press.dat"
    PRESS_POT = "press_pot.dat"
    BORN      = "born.dat"
    BORN_NL   = "born_nl.dat"
    SIM_LOG   = "sim.log"
    def __str__(self):
        return self.value

class FileKeys(Enum):
    THERMO    = "thermo"
    PRESS     = "press"
    PRESS_POT = "press_pot"
    BORN      = "born"
    BORN_NL   = "born_nl"

FILES = {
    FileKeys.THERMO.value   : Filenames.THERMO.value,
    FileKeys.PRESS.value    : Filenames.PRESS.value,
    FileKeys.PRESS_POT.value: Filenames.PRESS_POT.value,
    FileKeys.BORN.value     : Filenames.BORN.value,
    FileKeys.BORN_NL.value  : Filenames.BORN_NL.value
}

class Filepaths(Enum):
    ROOT   = Path('../').resolve()
    INPUTS = ROOT / 'input_scripts'
    DATA   = ROOT / 'data'
    

class Data:
    def __init__(self, data_path):
        self.path = Path(data_path)

    def read(self, dataset=None, dropdup=None, nrows=None, born_len=None):
        dfs = self.read_df(dataset, dropdup, nrows)
        return self.df2arr(dfs, born_len=born_len)
    
    def read_df(self, dataset=None, dropdup=None, nrows=None):
        if dataset is None:
            dataset = list(FILES.keys())
        # NOTE: This has to be consistent with the values to save in ScriptGenLJ.save_fixes()
        dataset_config = {
            FileKeys.THERMO.value: {
                'names': ['timestep', 'time', 'temp', 'energy', 'vol', 'dens'],
            },
            FileKeys.PRESS.value: {
                'names': ['timestep', 'time'] + ['xx','yy','zz','xy','xz','yz'],
            },
            FileKeys.PRESS_POT.value: {
                'names': ['timestep', 'time'] + ['xx','yy','zz','xy','xz','yz'],
            },
            FileKeys.BORN.value: {
                'names': ['timestep', 'time'] + ['p'+str(i) for i in range(1,22)],
            },
            FileKeys.BORN_NL.value: {
                'names': ['timestep', 'time'] + ['p'+str(i) for i in range(1,127)],
            },
        }
        result = {}
        for data in dataset:
            if data not in dataset_config:
                raise ValueError(f"Unknown dataset: {data}. Available: {list(dataset_config.keys())}")
            config = dataset_config[data]
            result[data] = pd.read_csv(
                self.path / FILES[data],
                sep=' ',
                header=None,
                names=config['names'],
                skiprows=2,
                nrows=nrows,
                index_col=False,
                comment='#')
            if dropdup:
                if dropdup == True:
                    result[data] = result[data].drop_duplicates()
                elif isinstance(dropdup, list):
                    result[data] = result[data].drop_duplicates(subset=dropdup)
                else:
                    raise ValueError("Incorrect 'dropdup' value.")
        return result 

    def df2arr(self, dfs, trim=None, born_len=None):
        #thermo_data, press_data, press_pot_data, born_data, born_nl_data = (
        #    dfs.values() + [None,]*5)[:5]
        result = {}
        if FileKeys.THERMO.value in dfs.keys():
            thermo_data = dfs[FileKeys.THERMO.value]
            V = thermo_data.vol.mean()
            result['V'] = V
            result['T'] = thermo_data.temp.mean()
            result['time'] = thermo_data.time

        scale = self.get_scale()
        if FileKeys.PRESS.value in dfs.keys():
            press_pot_data = dfs[FileKeys.PRESS.value]
            press_raw = np.array(press_pot_data.iloc[:trim,2:]) * scale
            stress = -lammps_array_to_matrix(press_raw)
            result['stress'] = stress
            #result['stress_d'] = stress - stress.mean(axis=0)
            result['press_avg'] = -np.trace(stress.mean(axis=0)) / 3
            
        if (FileKeys.PRESS_POT.value in dfs.keys() 
            and FileKeys.BORN.value in dfs.keys()):

            press_pot_data = dfs[FileKeys.PRESS_POT.value]
            born_len = min(born_len, len(press_pot_data)) if born_len else len(press_pot_data)
            press_pot_raw = np.array(press_pot_data.iloc[:born_len,2:]) * scale
            press_kin_raw = press_raw[:born_len] - press_pot_raw
            stress_kin = -lammps_array_to_matrix(press_kin_raw)
            stress_kin_mean = stress_kin.mean(axis=0)
    
            born_data = dfs[FileKeys.BORN.value]
            born_len = min(born_len, len(born_data))
            born_raw = np.array(born_data.iloc[:born_len,2:]) * scale / V
            born_voigt = lammps_array_to_matrix(born_raw)
            CB = voigt_to_tensor(born_voigt)
            CB_avg = CB.mean(axis=0)
            CBK = CB - 4 * sym_dyad(np.eye(3), stress_kin[:born_len])
            CBK_avg = CBK.mean(axis=0)
            result['CBK'] = CBK
            result['CBK_avg'] = CBK_avg
    
        if FileKeys.BORN_NL.value in dfs.keys():
            born_nl_data = dfs[FileKeys.BORN_NL.value]
            born_nl_raw = np.array(born_nl_data.iloc[:,2:]) * scale / V
            born_nl_voigt = lammps_array_to_matrix(born_nl_raw.reshape(
                len(born_nl_raw), 21, 6).swapaxes(-1, -2))
            NB_avg = voigt_to_tensor(born_nl_voigt, order=3).mean(axis=0)
            NK_avg = -sym_dyad(np.eye(3), CBK_avg) / 8
            NK_avg += (NK_avg.swapaxes(0, 2).swapaxes(1,3) + 
                       NK_avg.swapaxes(0, 4).swapaxes(1,5) )
            result['NBK_avg'] = NB_avg + NK_avg
    
        return result
    

    def get_scale(self):
        """ Lennard-Jones scale """
        for l in open(self.path / Filenames.SIM_LOG.value):
            if l.startswith('pair_coeff'):
                p = l.split();
                eps, sigma = float(p[3]), float(p[4])
                assert (eps>0 and sigma>0)
                return eps / sigma**3
        warnings.warn("No scale was found in the logs, returning 1")
        return 1

    def get_wall_time(self):
        for l in open(self.path / Filenames.SIM_LOG.value, 'r'):
            if l.startswith('Total wall time'):
                p = l.split()[-1]
                return np.array([int(p.split(':')[i]) for i in range(3)]) @ np.array([3600, 60, 1])
        warnings.warn("No total wall time was found in the logs, returning None")
        return None


# def load_data(path, read_born_nl=True, skip=0):
#     thermo_data = pd.read_csv(os.path.join(path, 'thermo.dat'), sep=' ', header=0, 
#                               names=['timestep', 'time', 'temp', 'energy', 'vol', 'dens'],
#                               skiprows=2+skip, index_col=False, comment='#')
    
#     press_data = pd.read_csv(os.path.join(path, 'press.dat'), sep=' ', header=0, 
#                              names=['timestep', 'time']+['xx','yy','zz','xy','xz','yz'],
#                              skiprows=2+skip, index_col=False, comment='#')
    
#     press_pot_data = pd.read_csv(os.path.join(path, 'press_pot.dat'), sep=' ', header=0, 
#                                  names=['timestep', 'time']+['xx','yy','zz','xy','xz','yz'],
#                                  skiprows=2+skip, index_col=False, comment='#')
    
#     born_data = pd.read_csv(os.path.join(path, 'born.dat'), sep=' ', header=0, 
#                             names=['timestep', 'time']+['p'+str(i) for i in range(1,22)],
#                             skiprows=2+skip, index_col=False, comment='#')
#     if read_born_nl:
#         born_nl_data = pd.read_csv(os.path.join(path, 'born_nl.dat'), sep=' ', header=None, 
#                                    names=['timestep', 'time']+['p'+str(i) for i in range(1,127)],
#                                    skiprows=2, index_col=False, comment='#')
#         return thermo_data, press_data, press_pot_data, born_data, born_nl_data

#     return thermo_data, press_data, press_pot_data, born_data