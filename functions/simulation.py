import math
import numpy as np
import matplotlib.pyplot as plt


class Drone:
    def __init__(self, config: dict):
        self.config = config

        # state variables
        self.t = None

        self.x = None
        self.y = None
        self.z = None

        self.vx = None
        self.vy = None
        self.vz = None

        self.ax = None
        self.ay = None
        self.az = None

        self.roll = None
        self.pitch = None
        self.yaw = None

        self.p = None
        self.q = None
        self.r = None

        self.pdot = None
        self.qdot = None
        self.rdot = None

        self.T1 = None
        self.T2 = None
        self.T3 = None
        self.T4 = None

        self.thrust = np.array([self.T1, self.T2, self.T3, self.T4])

        # Define parameters
        self.m = config['specs'].get('mass')  # Mass
        self.L = config['specs'].get('arm_length')  # Arm length
        self.Ixx = config['specs'].get('Ixx')  # Moment of inertia around x-axis
        self.Iyy = config['specs'].get('Iyy')  # Moment of inertia around y-axis
        self.Izz = config['specs'].get('Izz')  # Moment of inertia around z-axis
        self.k_drag_linear = config['specs'].get('k_drag_linear')  # Linear drag coefficient
        self.k_drag_angular = config['specs'].get('k_drag_angular')  # Angular drag coefficient
        self.g = config['physics'].get('gravitational_acceleration')  # Gravitational acceleration
        self.dt = config['system_defaults'].get('delta_t')  # s

        self.historian = {'t': [], 'x': [], 'y': [], 'z': [],
                          'T1': [], 'T2': [], 'T3': [], 'T4': [],
                          'ax': [], 'ay': [], 'az': [],
                          'vx': [], 'vy': [], 'vz': [],
                          'pdot': [], 'qdot': [], 'rdot': []}

    def equations_of_motion(self):
        # linear dynamics as a function of thrust
        self.ax = ((1 / self.m)
                   * (math.sin(self.roll) * math.cos(self.pitch) * math.sin(self.yaw)
                      + math.cos(self.roll) * math.sin(self.pitch) * math.cos(self.yaw))
                   * (self.T1 + self.T2 + self.T3 + self.T4)
                   - self.k_drag_linear * self.vx
                   )

        self.ay = ((1 / self.m)
                   * (-math.cos(self.roll) * math.cos(self.pitch) * math.sin(self.yaw)
                      + math.sin(self.roll) * math.sin(self.pitch) * math.cos(self.yaw))
                   * (self.T1 + self.T2 + self.T3 + self.T4)
                   - self.k_drag_linear * self.vy
                   )

        self.az = ((1 / self.m)
                   * -math.cos(self.roll) * math.cos(self.pitch)
                   * (self.T1 + self.T2 + self.T3 + self.T4)
                   - self.k_drag_linear * self.vz
                   + self.g
                   )

        # angular dynamics
        self.pdot = ((1 / self.Ixx)
                     * self.L
                     * (self.T2 - self.T4)
                     - self.k_drag_angular * self.p
                     )

        self.qdot = ((1 / self.Iyy)
                     * self.L
                     * (self.T3 - self.T1)
                     - self.k_drag_angular * self.q
                     )

        self.rdot = ((1 / self.Izz)
                     * self.L
                     * (self.T1 - self.T2 + self.T3 - self.T4)
                     - self.k_drag_angular * self.r
                     )

    def derivatives(self):
        # time
        self.t = self.t + self.dt

        # linear velocity derivatives
        self.vx = self.ax * self.dt + self.vx
        self.vy = self.ay * self.dt + self.vy
        self.vz = self.az * self.dt + self.vz

        # position derivatives
        self.x = self.vx * self.dt + self.x
        self.y = self.vy * self.dt + self.y
        self.z = self.vz * self.dt + self.z

        # angular velocity derivatives
        self.p = self.pdot * self.dt + self.p
        self.q = self.qdot * self.dt + self.q
        self.r = self.rdot * self.dt + self.r

        # orientation derivatives
        self.roll = (self.p
                     + math.tan(self.pitch) * (self.q * math.sin(self.roll)
                                               + self.r * math.cos(self.roll))
                     ) + self.roll

        self.pitch = (self.q * math.cos(self.roll)
                      - self.r * math.sin(self.roll)) + self.pitch

        self.yaw = (self.q * math.sin(self.roll)
                    + self.r * math.cos(self.roll)) / math.cos(self.pitch) + self.yaw

    def initialize(self):
        print('initializing model')
        self.t = 0

        self.x = 0
        self.y = 0
        self.z = 0

        self.vx = 0
        self.vy = 0
        self.vz = 0

        self.ax = 0
        self.ay = 0
        self.az = 0

        self.roll = 0
        self.pitch = 0
        self.yaw = 0

        self.p = 0
        self.q = 0
        self.r = 0

        self.pdot = 0
        self.qdot = 0
        self.rdot = 0

        self.historian = {'t': [], 'x': [], 'y': [], 'z': [], 'T1': [], 'T2': [], 'T3': [], 'T4': [],
                          'ax': [], 'ay': [], 'az': [], 'vx': [], 'vy': [], 'vz': [],
                          'pdot': [], 'qdot': [], 'rdot': [],
                          'p': [], 'q': [], 'r': []}

    def update_thrust(self):
        ss_thrust = 0.2455
        stabilizer_factor = 1
        self.T1 = ss_thrust * stabilizer_factor * (1 + math.sin(self.t * math.pi) / 100)
        self.T2 = ss_thrust * stabilizer_factor
        self.T3 = ss_thrust * stabilizer_factor
        self.T4 = ss_thrust * stabilizer_factor

        self.T1 = 1 * math.sin(self.t * math.pi)
        self.T2 = 1 * math.sin(self.t * math.pi + math.pi / 5)
        self.T3 = 1 * math.sin(self.t * math.pi + 2 * math.pi / 5)
        self.T4 = 1 * math.sin(self.t * math.pi + 3 * math.pi / 5)

        self.thrust = np.array([self.T1, self.T2, self.T3, self.T4])

    def record_data(self):
        self.historian['t'].append(self.t)
        self.historian['x'].append(self.x)
        self.historian['y'].append(self.y)
        self.historian['z'].append(self.z)
        self.historian['T1'].append(self.T1)
        self.historian['T2'].append(self.T2)
        self.historian['T3'].append(self.T3)
        self.historian['T4'].append(self.T4)
        self.historian['ax'].append(self.x)
        self.historian['ay'].append(self.y)
        self.historian['az'].append(self.z)
        self.historian['vx'].append(self.x)
        self.historian['vy'].append(self.y)
        self.historian['vz'].append(self.z)
        self.historian['pdot'].append(self.pdot)
        self.historian['qdot'].append(self.qdot)
        self.historian['rdot'].append(self.rdot)
        self.historian['p'].append(self.p)
        self.historian['q'].append(self.q)
        self.historian['r'].append(self.r)

    def plot(self):
        fig, ax = plt.subplots(7, 1)
        ax[0].plot(self.historian['t'], self.historian['x'])
        ax[1].plot(self.historian['t'], self.historian['y'])
        ax[2].plot(self.historian['t'], self.historian['z'])
        ax[3].plot(self.historian['t'], self.historian['T1'])
        ax[4].plot(self.historian['t'], self.historian['T2'])
        ax[5].plot(self.historian['t'], self.historian['T3'])
        ax[6].plot(self.historian['t'], self.historian['T4'])

        fig.savefig(
            "C:\\Users\\baris\\Documents\\Python Scripts\\projects\\database\\drone\\output\\detailed_linearprofile.png",
            dpi=300)

        fig, ax = plt.subplots(7, 1)
        ax[0].plot(self.historian['t'], self.historian['p'])
        ax[1].plot(self.historian['t'], self.historian['q'])
        ax[2].plot(self.historian['t'], self.historian['r'])
        ax[3].plot(self.historian['t'], self.historian['T1'])
        ax[4].plot(self.historian['t'], self.historian['T2'])
        ax[5].plot(self.historian['t'], self.historian['T3'])
        ax[6].plot(self.historian['t'], self.historian['T4'])

        fig.savefig(
            "C:\\Users\\baris\\Documents\\Python Scripts\\projects\\database\\drone\\output\\detailed_angular_profile.png",
            dpi=300)

        fig, ax = plt.subplots()
        ax3d = plt.axes(projection='3d')
        ax3d.scatter3D(self.historian['x'], self.historian['y'], self.historian['z'])
        fig.savefig("C:\\Users\\baris\\Documents\\Python Scripts\\projects\\database\\drone\\output\\projectile_3d.png",
                    dpi=300)

    def simulate(self):
        self.initialize()
        while self.t < 10:
            # print(f"Time: {self.t:.1f}\t||\t Coordinates:\t{self.x:.1f}\t|\t {self.y:.1f}\t|\t{self.z:.1f}")
            print(f"Time: {self.t:.1f}\t||\t Acceleration:\t{self.ax:.1f}\t|\t {self.ay:.1f}\t|\t{self.az:.3f}")
            self.step()
            self.record_data()

    def step(self):
        # print(f"Time: {self.t:.1f}\t||\t Coordinates:\t{self.x:.1f}\t|\t {self.y:.1f}\t|\t{self.z:.1f}")
        print(f"Time: {self.t:.1f}\t||\t Acceleration:\t{self.ax:.1f}\t|\t {self.ay:.1f}\t|\t{self.az:.3f}")
        self.update_thrust()
        self.equations_of_motion()
        self.derivatives()