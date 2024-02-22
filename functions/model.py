import math
import matplotlib.pyplot as plt
import numpy as np
import pyomo.environ as pyo
import pyomo.dae as pyd


class Drone:
    def __init__(self):
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
        self.m = 0.1  # Mass
        self.L = 0.25  # Arm length
        self.Ixx = 0.1  # Moment of inertia around x-axis
        self.Iyy = 0.1  # Moment of inertia around y-axis
        self.Izz = 0.2  # Moment of inertia around z-axis
        self.k_drag_linear = 0.1  # Linear drag coefficient
        self.k_drag_angular = 0.05  # Angular drag coefficient
        self.g = 9.81  # Gravitational acceleration
        self.dt = 0.1  # s

        self.historian = {'t':[], 'x': [], 'y': [], 'z': [],
                          'T1': [], 'T2':[], 'T3': [], 'T4': [],
                          'ax': [], 'ay': [], 'az': [],
                          'vx': [], 'vy': [], 'vz': [],
                          'pdot': [], 'qdot': [], 'rdot': []}

    def equations_of_motion(self):
        # linear dynamics as a function of thrust
        self.ax = ((1/self.m)
                   * (math.sin(self.roll) * math.cos(self.pitch) * math.sin(self.yaw)
                      + math.cos(self.roll) * math.sin(self.pitch) * math.cos(self.yaw))
                   * (self.T1 + self.T2 + self.T3 + self.T4)
                   - self.k_drag_linear * self.vx
                   )

        self.ay = ((1/self.m)
                   * (-math.cos(self.roll) * math.cos(self.pitch) * math.sin(self.yaw)
                      + math.sin(self.roll) * math.sin(self.pitch) * math.cos(self.yaw))
                   * (self.T1 + self.T2 + self.T3 + self.T4)
                   - self.k_drag_linear * self.vy
                   )

        self.az = ((1/self.m)
                   * -math.cos(self.roll) * math.cos(self.pitch)
                   * (self.T1 + self.T2 + self.T3 + self.T4)
                   - self.k_drag_linear * self.vz
                   + self.g
                   )

        # angular dynamics
        self.pdot = ((1/self.Ixx)
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
                      -self.r * math.sin(self.roll)) + self.pitch

        self.yaw = (self.q * math.sin(self.roll)
                    + self.r * math.cos(self.roll))/math.cos(self.pitch) + self.yaw

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

        self.historian = {'t':[], 'x': [], 'y': [], 'z': [], 'T1': [], 'T2':[], 'T3': [], 'T4': [],
                          'ax': [], 'ay': [], 'az': [], 'vx': [], 'vy': [], 'vz': [],
                          'pdot': [], 'qdot': [], 'rdot': [],
                          'p': [], 'q': [], 'r': []}


    def update_thrust(self):
        ss_thrust = 0.2455
        stabilizer_factor = 1
        self.T1 = ss_thrust * stabilizer_factor * (1 + math.sin(self.t * math.pi)/100)
        self.T2 = ss_thrust * stabilizer_factor
        self.T3 = ss_thrust * stabilizer_factor
        self.T4 = ss_thrust * stabilizer_factor

        # self.T1 = 1*math.sin(self.t * math.pi)
        # self.T2 = 1*math.sin(self.t * math.pi + math.pi/5)
        # self.T3 = 1*math.sin(self.t * math.pi + 2*math.pi/5)
        # self.T4 = 1*math.sin(self.t * math.pi + 3*math.pi/5)

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

        fig.savefig("C:\\Users\\baris\\Documents\\Python Scripts\\projects\\database\\drone\\output\\detailed_linearprofile.png", dpi=300)

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
        fig.savefig("C:\\Users\\baris\\Documents\\Python Scripts\\projects\\database\\drone\\output\\projectile_3d.png", dpi=300)

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


class DronePyomo:
    def __init__(self):
        self.model = pyo.AbstractModel()

        self.model.t = pyd.ContinuousSet(bounds=(0, 1))
        self.define_vars()
        self.define_derivatives()
        self.define_parameters()
        self.define_constraints()

    def define_vars(self):
        print('Defining variables')
        self.model.time = pyo.Var()

        # Cartesian coordinates of location
        self.model.x = pyo.Var(self.model.t)
        self.model.y = pyo.Var(self.model.t)
        self.model.z = pyo.Var(self.model.t)

        # Angular position
        self.model.roll = pyo.Var(self.model.t)
        self.model.pitch = pyo.Var(self.model.t)
        self.model.yaw = pyo.Var(self.model.t)

        # Angular velocity
        self.model.p = pyo.Var(self.model.t)
        self.model.q = pyo.Var(self.model.t)
        self.model.r = pyo.Var(self.model.t)

        # Thrust
        self.model.T1 = pyo.Var(self.model.t)
        self.model.T2 = pyo.Var(self.model.t)
        self.model.T3 = pyo.Var(self.model.t)
        self.model.T4 = pyo.Var(self.model.t)

    def define_derivatives(self):
        print('Defining derivative variables')

        # Velocity
        self.model.vx = pyd.DerivativeVar(self.model.x, wrt=self.model.t)
        self.model.vy = pyd.DerivativeVar(self.model.y, wrt=self.model.t)
        self.model.vz = pyd.DerivativeVar(self.model.z, wrt=self.model.t)

        # Acceleration
        self.model.ax = pyd.DerivativeVar(self.model.vx, wrt=self.model.t)
        self.model.ay = pyd.DerivativeVar(self.model.vy, wrt=self.model.t)
        self.model.az = pyd.DerivativeVar(self.model.vz, wrt=self.model.t)

        # Angular acceleration
        self.model.pdot = pyd.DerivativeVar(self.model.p, wrt=self.model.t)
        self.model.qdot = pyd.DerivativeVar(self.model.q, wrt=self.model.t)
        self.model.rdot = pyd.DerivativeVar(self.model.r, wrt=self.model.t)

    def define_parameters(self):
        print('Defining parameters')
        self.model.m = 0.1  # Mass
        self.model.L = 0.25  # Arm length
        self.model.Ixx = 0.1  # Moment of inertia around x-axis
        self.model.Iyy = 0.1  # Moment of inertia around y-axis
        self.model.Izz = 0.2  # Moment of inertia around z-axis
        self.model.k_drag_linear = 0.1  # Linear drag coefficient
        self.model.k_drag_angular = 0.05  # Angular drag coefficient
        self.model.g = 9.81  # Gravitational acceleration

    def cons_linear_acceleration_x(self):
        return self.model.ax == ((1/self.model.m)
                                 * (pyo.sin(self.model.roll)*pyo.cos(self.model.pitch)*pyo.sin(self.model.yaw)
                                    + pyo.cos(self.model.roll)*pyo.sin(self.model.pitch)*pyo.cos(self.model.yaw))
                                 * (self.model.T1 + self.model.T2 + self.model.T3 + self.model.T4)
                                 - self.model.K_drag_linear * self.model.vx
                                 ) * self.model.time

    def cons_linear_acceleration_y(self):
        return self.model.ay == ((1/self.model.m)
                                 * (-pyo.cos(self.model.roll) * pyo.cos(self.model.pitch) * pyo.sin(self.model.yaw)
                                    + pyo.sin(self.model.roll) * pyo.sin(self.model.pitch) * pyo.cos(self.model.yaw))
                                 * (self.model.T1 + self.model.T2 + self.model.T3 + self.model.T4)
                                 - self.model.k_drag_linear * self.model.vy
                                 ) * self.model.time

    def cons_linear_acceleration_z(self):
        return self.model.az == ((1/self.model.m)
                                 * -pyo.cos(self.model.roll) * pyo.cos(self.model.pitch)
                                 * (self.model.T1 + self.model.T2 + self.model.T3 + self.model.T4)
                                 - self.model.k_drag_linear * self.model.vz
                                 + self.model.g
                                 ) * self.model.time

    def cons_angular_dynamics_pdot(self):
        return self.model.pdot == ((1/self.model.Ixx)
                                   * self.model.L
                                   * (self.model.T2 - self.model.T4)
                                   - self.model.k_drag_angular * self.model.p
                                   ) * self.model.time

    def cons_angular_dynamics_qdot(self):
        return self.model.qdot == ((1 / self.model.Iyy)
                                   * self.model.L
                                   * (self.model.T3 - self.model.T1)
                                   - self.model.k_drag_angular * self.model.q
                                   ) * self.model.time

    def cons_angular_dynamics_rdot(self):
        return self.model.rdot == ((1 / self.model.Izz)
                                   * self.model.L
                                   * (self.model.T1 - self.model.T2 + self.model.T3 - self.model.T4)
                                   - self.model.k_drag_angular * self.model.r
                                   ) * self.model.time

    def equations_of_motion(self):
        print('Defining equations of motion')

        # linear dynamics
        self.model.cons_linear_acceleration_x = pyo.Constraint(rule=self.cons_linear_acceleration_x)
        self.model.cons_linear_acceleration_y = pyo.Constraint(rule=self.cons_linear_acceleration_y)
        self.model.cons_linear_acceleration_z = pyo.Constraint(rule=self.cons_linear_acceleration_z)

        # angular dynamics
        self.model.cons_angular_dynamics_pdot = pyo.Constraint(rule=self.cons_angular_dynamics_pdot)
        self.model.cons_angular_dynamics_qdot = pyo.Constraint(rule=self.cons_angular_dynamics_qdot)
        self.model.cons_angular_dynamics_rdot = pyo.Constraint(rule=self.cons_angular_dynamics_rdot)

    def terminal_constraints(self):
        print('Adding steady state terminal constraints.')
        # self.model.terminal_constraints = pyo.ConstraintList()
        #
        # # Velocity
        # self.model.terminal_constraints.add(expr=self.model.vx[1] == 0)
        # self.model.terminal_constraints.add(expr=self.model.vy[1] == 0)
        # self.model.terminal_constraints.add(expr=self.model.vz[1] == 0)
        #
        # # Acceleration
        # self.model.terminal_constraints.add(expr=self.model.ax[1] == 0)
        # self.model.terminal_constraints.add(expr=self.model.ay[1] == 0)
        # self.model.terminal_constraints.add(expr=self.model.az[1] == 0)
        #
        # # Angular acceleration
        # self.model.terminal_constraints.add(expr=self.model.pdot[1] == 0)
        # self.model.terminal_constraints.add(expr=self.model.qdot[1] == 0)
        # self.model.terminal_constraints.add(expr=self.model.rdot[1] == 0)
        #
        # # Angular velocity
        # self.model.terminal_constraints.add(expr=self.model.p[1] == 0)
        # self.model.terminal_constraints.add(expr=self.model.q[1] == 0)
        # self.model.terminal_constraints.add(expr=self.model.r[1] == 0)

    def obj_minimize_time(self):
        print('Objective: minimize time')
        return self.model.time

    def define_constraints(self):
        print('Defining constraints')
        self.equations_of_motion()
        self.terminal_constraints()

    def define_objective(self):
        print('Defining objective function')
        self.model.obj_minimize_time = pyo.Objective(rule=self.obj_minimize_time())
