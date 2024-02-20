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
        self.state_space_identified = None
        self.covariance_matrix = None
        self.kalman_filter = None
        self.system_input = None
        self.system_output = None

    def update_historian(self):
        self.historian = pd.DataFrame.from_dict(self.drone.historian, orient='columns')

    def collect_data(self):
        self.drone.simulate()
        self.update_historian()
        print(self.historian)

    def perform_subspace_identification(self):
        self.n4sid = NFourSID(
            self.historian,
            output_columns=['x', 'y', 'z'],
            input_columns=['T1', 'T2', 'T3', 'T4'],
            num_block_rows=5
        )
        # output_columns=['x', 'y', 'z', 'vx', 'vy', 'vz', 'ax', 'ay', 'az', 'pdot', 'qdot', 'rdot'],
        self.n4sid.subspace_identification()
        self.state_space_identified, self.covariance_matrix = self.n4sid.system_identification(rank=3)

    def get_kalman_filter(self):
        self.kalman_filter = Kalman(self.state_space_identified, self.covariance_matrix)

    def get_io_from_model(self):
        self.system_input = self.drone.thrust.reshape((4, 1))
        # self.system_output = np.array([self.drone.x, self.drone.y, self.drone.z,
        #                                self.drone.vx, self.drone.vy, self.drone.vz,
        #                                self.drone.ax, self.drone.ay, self.drone.az,
        #                                self.drone.pdot, self.drone.qdot, self.drone.rdot]).reshape((12, 1))
        self.system_output = np.array([self.drone.x, self.drone.y, self.drone.z]).reshape((3, 1))

    def simulate_kalman_filter(self):
        # should separate training and testing instances for the drones
        self.drone.initialize()
        while self.drone.t < 5:
            print(f"Time: {self.drone.t:.1f}\t||\t Coordinates:\t{self.drone.x:.1f}\t|\t {self.drone.y:.1f}\t|\t{self.drone.z:.1f}")
            self.drone.step()
            self.get_io_from_model()
            self.kalman_filter.step(self.system_output, self.system_input)
        self.kalman_filter.to_dataframe().to_csv("C:\\Users\\baris\\Documents\\Python Scripts\\projects\\database\\drone\\output\\kalman_filter.csv")

    def plot_eigenvectors(self):
        fig, ax = plt.subplots()
        self.n4sid.plot_eigenvalues(ax)
        fig.savefig(
            "C:\\Users\\baris\\Documents\\Python Scripts\\projects\\database\\drone\\output\\ssi_eigenvalues.png",
            dpi=300)

    def plot_kalman_simulation(self):
        fig, ax = plt.subplots()
        self.kalman_filter.plot_filtered(fig)
        fig.savefig("C:\\Users\\baris\\Documents\\Python Scripts\\projects\\database\\drone\\output\\kalman_simulation.png",
                    dpi=300)

    def run(self):
        self.collect_data()
        self.perform_subspace_identification()
        self.plot_eigenvectors()
        self.get_kalman_filter()
        self.simulate_kalman_filter()
        self.plot_kalman_simulation()

