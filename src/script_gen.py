from pathlib import Path
from .read_write import FILES

class ScriptGenLJ:
    def __init__(self, savedir):
        self.path = Path(savedir)

    def log(self, custom_path=None, append=False):
        if custom_path == None:
            custom_path = f"{self.path / 'sim.log'}"
        res = f"log {custom_path} "
        if append:
            res += "append"
        return res + "\n"
        
    def init_box_atoms(self, rho=1.058, L=10, T=4.0, seed=94673, offset = 0.05,
                       Rin=2.5, Rc=3.5, eps=1.0, sigma=1.0):
        self.T = T
        if Rc < Rin:
            raise ValueError("Rin should be less than Rc")
        return f"""\
# Initialization
units lj
dimension 3
boundary p p p
atom_style atomic

# Create simulation box and atoms
lattice fcc {rho} origin {offset} {offset} {offset}
region simbox block 0 {L} 0 {L} 0 {L} units lattice
create_box 1 simbox
create_atoms 1 box
mass 1 1.0
velocity all create {T} {seed} mom yes

# Potential settings
pair_style lj/smooth {Rin} {Rc}
pair_coeff 1 1 {eps} {sigma}
pair_modify shift yes
"""
        
    def computes(self, comp_born=True, comp_born_nl=False, numdiff=1e-6):
        self.comp_born = comp_born
        self.comp_born_nl = comp_born_nl
        script = f"""
# Monitoring variables
variable time equal time
variable etot equal etotal
variable vol equal vol
variable dens equal density

# Default computes
compute temp all temp
compute press all pressure thermo_temp
"""
        if comp_born:
            script += f"""
# Born matrix computation
compute press_pot all pressure NULL virial
compute born_matrix all born/matrix numdiff {numdiff} press_pot
"""
            if comp_born_nl:
                script += f"compute born_matrix_nl all born/matrix/nonlinear \
                    {numdiff} press_pot born_matrix"
        return script

    def time_step_log(self, time_step=0.005, thermo_log_steps=5000):
        self.time_step = time_step
        return f"""
# Timestep (default 0.005 for lennard-jones)
timestep {time_step}

# Screen and log output
thermo {thermo_log_steps}
thermo_style custom step temp press etotal
thermo_modify flush yes
"""

    def run_equilibration(self, steps: int, ens='nvt', Tend=None, thermo_tau=None,
                          pstart=0.0, pend=0.0, baro_tau=None):
        Tend = self.T if Tend==None else Tend
        thermo_tau = 100*self.time_step if thermo_tau==None else thermo_tau
        baro_tau = 1000*self.time_step if baro_tau==None else baro_tau
        if ens == 'nvt':
            return f"""
fix nvt_eq all nvt temp {self.T} {Tend} {thermo_tau}
run {steps}
unfix nvt_eq
"""
        elif ens == 'npt':
            return f"""
fix npt_eq all npt temp {self.T} {self.T} {thermo_tau} iso {pstart} {pend} {baro_tau}
run {steps}
unfix npt_eq
"""    
        else:
            raise ValueError(f"Ensemble {ens} is not implemented")

    def run_production(self, steps: int, fix=True):
        if fix:
            return f"\nfix nve_prod all nve \nrun {steps}\n"
        else:
            return f"\nrun {steps}\n"

    def save_fixes(self, fix_params: dict, append=True):
        file_command = 'append' if append else 'file'
        # NOTE: This has to be consistent with the values to read in Data.read()
        config = {'thermo'   : 'v_time c_thermo_temp v_etot v_vol v_dens',
                  'press'    : 'v_time c_press[*]',
                  'press_pot': 'v_time c_press_pot[*]',
                  'born'     : 'v_time c_born_matrix[*]',
                  'born_nl'  : 'v_time c_born_matrix_nl[*]'}
        script = "\n# Output values\n"
        for fix, save_every in fix_params.items():
            script += f"""fix {fix}_output all ave/time 1 1 {save_every} {config[fix]} &
    {file_command} "{self.path / FILES[fix]}"\n"""
        return script

    def unfix_save(self, keys):
        script = "\n"
        for key in keys:
            if key in FILES.keys():
                script += f"unfix {key}_output\n"
            else:
                raise ValueError(f"Unknown key {key}")
        return script

    def write(self, name, restart=True):
        if restart:
            return f"""\nwrite_restart "{self.path / name}.restart"\n"""
        return f"""\nwrite_data "{self.path / name}.dump" nocoeff\n"""

    def read(self, name, restart=True):
        if restart:
            return f"""\nread_restart "{self.path / name}.restart"\n"""
        #return f"""\nwrite_data "{self.path / name}.dump" nocoeff\n"""
        raise ValueError("Restart from nonbinary file is not implemented.")