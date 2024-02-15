import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from nfoursid.kalman import Kalman
from nfoursid.nfoursid import NFourSID
from nfoursid.state_space import StateSpace


class SystemIdentification:
    def __init__(self, drone):
        self.drone = drone
        self.historian = pd.DataFrame()
        self.n4sid = None

    def update_historian(self):
        self.historian = pd.DataFrame.from_dict(self.drone.historian, orient='columns')

    def collect_data(self):
        self.drone.simulate()
        self.update_historian()
        print(self.historian)

    def perform_ssi(self):
        self.n4sid = NFourSID(
            self.historian,
            output_columns=['x', 'y', 'z', 'vx', 'vy', 'vz', 'ax', 'ay', 'az'],
            input_columns=['T1', 'T2', 'T3', 'T4'],
            num_block_rows=5
        )
        self.n4sid.subspace_identification()

    def plot_eigenvectors(self):
        fig, ax = plt.subplots()
        self.n4sid.plot_eigenvalues(ax)
        fig.savefig(
            "C:\\Users\\baris\\Documents\\Python Scripts\\projects\\database\\drone\\output\\ssi_eigenvalues.png",
            dpi=300)


    def run(self):
        self.collect_data()
        self.perform_ssi()
        self.plot_eigenvectors()
