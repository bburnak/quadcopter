import math


class Drone:
    def __init__(self):
        # state variables
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

        # Define parameters
        self.m = 0.1  # Mass
        self.L = 0.25  # Arm length
        self.Ixx = 0.1  # Moment of inertia around x-axis
        self.Iyy = 0.1  # Moment of inertia around y-axis
        self.Izz = 0.2  # Moment of inertia around z-axis
        self.k_drag_linear = 0.1  # Linear drag coefficient
        self.k_drag_angular = 0.05  # Angular drag coefficient
        self.g = 9.81  # Gravitational acceleration
        self.dt = 0.1 # s

        # initial conditions
        self.x_i = None
        self.y_i = None
        self.z_i = None

        self.vx_i = None
        self.vy_i = None
        self.vz_i = None

        self.p_i = None
        self.q_i = None
        self.r_i = None

        self.roll_i = None
        self.pitch_i = None
        self.yaw_i = None

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
                   (self.T1 + self.T2 + self.T3 + self.T4)
                   - self.k_drag_linear * self.vz
                   + self.g
                   )

        # angular dynamics
        self.pdot = ((1/self.Ixx)
                     * self.L
                     * (self.T2 - self.T4)
                     * self.k_drag_angular * self.p
                     )
        self.qdot = ((1 / self.Iyy)
                     * self.L
                     * (self.T3 - self.T1)
                     * self.k_drag_angular * self.q
                     )

        self.rdot = ((1 / self.Izz)
                     * self.L
                     * (self.T1 - self.T2 + self.T3 - self.T4)
                     * self.k_drag_angular * self.r
                     )

    def derivatives(self):
        # linear velocity derivatives
        self.vx = self.ax * self.dt + self.vx_i
        self.vy = self.ay * self.dt + self.vy_i
        self.vz = self.az * self.dt + self.vz_i

        # position derivatives
        self.x = self.vx * self.dt + self.x_i
        self.y = self.vy * self.dt + self.y_i
        self.z = self.vz * self.dt + self.z_i

        # angular velocity derivatives
        self.p = self.pdot * self.dt + self.p_i
        self.q = self.qdot * self.dt + self.q_i
        self.r = self.rdot * self.dt + self.r_i

        # orientation derivatives
        self.roll = (self.p
                     + math.tan(self.pitch) * (self.q * math.sin(self.roll)
                                               + self.r * math.cos(self.roll))
                     ) + self.roll_i

        self.pitch = (self.q * math.cos(self.roll)
                      -self.r * math.sin(self.roll)) + self.pitch_i

        self.yaw = (self.q * math.sin(self.roll)
                    + self.r * math.cos(self.roll))/math.cos(self.pitch) + self.yaw_i





    def print_confirm(self):
        print('model constructed')


