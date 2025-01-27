import os
import matplotlib.pyplot as plt
import pyomo.environ as pyo
import pyomo.dae as pyd


class DronePyomo:
    def __init__(self, config):
        self.config = config
        self.model = pyo.AbstractModel()
        self.instance = None
        self.results = None
        self.historian = None

        self.define_sets()
        self.define_vars()
        self.define_derivatives()
        self.define_parameters()
        self.define_constraints()
        self.define_objective()

    def define_sets(self):
        print('Defining sets')
        self.model.t = pyd.ContinuousSet(bounds=(0, 1))

    def define_vars(self):
        print('Defining variables')
        self.model.time = pyo.Var(within=pyo.NonNegativeReals)

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
        min_thrust = self.config['specs']['thrust']['min']
        max_thrust = self.config['specs']['thrust']['max']
        self.model.T1 = pyo.Var(self.model.t, bounds=(min_thrust, max_thrust))
        self.model.T2 = pyo.Var(self.model.t, bounds=(min_thrust, max_thrust))
        self.model.T3 = pyo.Var(self.model.t, bounds=(min_thrust, max_thrust))
        self.model.T4 = pyo.Var(self.model.t, bounds=(min_thrust, max_thrust))

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
        self.model.m = pyo.Param(initialize=0.1)  # Mass
        self.model.L = pyo.Param(initialize=0.25)  # Arm length
        self.model.Ixx = pyo.Param(initialize=0.1)  # Moment of inertia around x-axis
        self.model.Iyy = pyo.Param(initialize=0.1)  # Moment of inertia around y-axis
        self.model.Izz = pyo.Param(initialize=0.2)  # Moment of inertia around z-axis
        self.model.k_drag_linear = pyo.Param(initialize=0.1)  # Linear drag coefficient
        self.model.k_drag_angular = pyo.Param(initialize=0.05)  # Angular drag coefficient
        self.model.g = pyo.Param(initialize=9.81)  # Gravitational acceleration

    def equations_of_motion(self):
        print('Defining equations of motion')

        def cons_linear_acceleration_x(model, t):
            if t == 0:
                return pyo.Constraint.Skip
            return model.ax[t] == ((1 / model.m)
                                   * (pyo.sin(model.roll[t]) * pyo.cos(model.pitch[t]) * pyo.sin(model.yaw[t])
                                      + pyo.cos(model.roll[t]) * pyo.sin(model.pitch[t]) * pyo.cos(model.yaw[t]))
                                   * (model.T1[t] + model.T2[t] + model.T3[t] + model.T4[t])
                                   - model.k_drag_linear * model.vx[t]
                                   ) * model.time

        def cons_linear_acceleration_y(model, t):
            if t == 0:
                return pyo.Constraint.Skip
            return model.ay[t] == ((1 / model.m)
                                   * (-pyo.cos(model.roll[t]) * pyo.cos(model.pitch[t]) * pyo.sin(model.yaw[t])
                                      + pyo.sin(model.roll[t]) * pyo.sin(model.pitch[t]) * pyo.cos(model.yaw[t]))
                                   * (model.T1[t] + model.T2[t] + model.T3[t] + model.T4[t])
                                   - model.k_drag_linear * model.vy[t]
                                   ) * model.time

        def cons_linear_acceleration_z(model, t):
            if t == 0:
                return pyo.Constraint.Skip
            return model.az[t] == ((1 / model.m)
                                   * -pyo.cos(model.roll[t]) * pyo.cos(model.pitch[t])
                                   * (model.T1[t] + model.T2[t] + model.T3[t] + model.T4[t])
                                   - model.k_drag_linear * model.vz[t]
                                   + model.g
                                   ) * model.time

        def cons_angular_dynamics_pdot(model, t):
            if t == 0:
                return pyo.Constraint.Skip
            return model.pdot[t] == ((1 / model.Ixx)
                                     * model.L
                                     * (model.T2[t] - model.T4[t])
                                     - model.k_drag_angular * model.p[t]
                                     ) * model.time

        def cons_angular_dynamics_qdot(model, t):
            if t == 0:
                return pyo.Constraint.Skip
            return model.qdot[t] == ((1 / model.Iyy)
                                     * model.L
                                     * (model.T3[t] - model.T1[t])
                                     - model.k_drag_angular * model.q[t]
                                     ) * model.time

        def cons_angular_dynamics_rdot(model, t):
            if t == 0:
                return pyo.Constraint.Skip
            return model.rdot[t] == ((1 / model.Izz)
                                     * model.L
                                     * (model.T1[t] - model.T2[t] + model.T3[t] - model.T4[t])
                                     - model.k_drag_angular * model.r[t]
                                     ) * model.time

        # linear dynamics
        self.model.cons_linear_acceleration_x = pyo.Constraint(self.model.t, rule=cons_linear_acceleration_x)
        self.model.cons_linear_acceleration_y = pyo.Constraint(self.model.t, rule=cons_linear_acceleration_y)
        self.model.cons_linear_acceleration_z = pyo.Constraint(self.model.t, rule=cons_linear_acceleration_z)

        # angular dynamics
        self.model.cons_angular_dynamics_pdot = pyo.Constraint(self.model.t, rule=cons_angular_dynamics_pdot)
        self.model.cons_angular_dynamics_qdot = pyo.Constraint(self.model.t, rule=cons_angular_dynamics_qdot)
        self.model.cons_angular_dynamics_rdot = pyo.Constraint(self.model.t, rule=cons_angular_dynamics_rdot)

    def terminal_constraints(self):
        print('Adding steady state terminal constraints.')

        def terminal_constraints_x(model, t):
            if t == 1:
                return model.x[t] == 0
            else:
                return pyo.Constraint.Skip

        def terminal_constraints_y(model, t):
            if t == 1:
                return model.y[t] == 0
            else:
                return pyo.Constraint.Skip

        def terminal_constraints_z(model, t):
            if t == 1:
                return model.z[t] == 0
            else:
                return pyo.Constraint.Skip

        def terminal_constraints_vx(model, t):
            if t == 1:
                return model.vx[t] == 0
            else:
                return pyo.Constraint.Skip

        def terminal_constraints_vy(model, t):
            if t == 1:
                return model.vy[t] == 0
            else:
                return pyo.Constraint.Skip

        def terminal_constraints_vz(model, t):
            if t == 1:
                return model.vz[t] == 0
            else:
                return pyo.Constraint.Skip

        def terminal_constraints_ax(model, t):
            if t == 1:
                return model.ax[t] == 0
            else:
                return pyo.Constraint.Skip

        def terminal_constraints_ay(model, t):
            if t == 1:
                return model.ay[t] == 0
            else:
                return pyo.Constraint.Skip

        def terminal_constraints_az(model, t):
            if t == 1:
                return model.az[t] == 0
            else:
                return pyo.Constraint.Skip

        def terminal_constraints_pdot(model, t):
            if t == 1:
                return model.pdot[t] == 0
            else:
                return pyo.Constraint.Skip

        def terminal_constraints_qdot(model, t):
            if t == 1:
                return model.qdot[t] == 0
            else:
                return pyo.Constraint.Skip

        def terminal_constraints_rdot(model, t):
            if t == 1:
                return model.rdot[t] == 0
            else:
                return pyo.Constraint.Skip

        def terminal_constraints_p(model, t):
            if t == 1:
                return model.p[t] == 0
            else:
                return pyo.Constraint.Skip

        def terminal_constraints_q(model, t):
            if t == 1:
                return model.q[t] == 0
            else:
                return pyo.Constraint.Skip

        def terminal_constraints_r(model, t):
            if t == 1:
                return model.r[t] == 0
            else:
                return pyo.Constraint.Skip

        # Position
        self.model.terminal_constraints_x = pyo.Constraint(self.model.t, rule=terminal_constraints_x)
        self.model.terminal_constraints_y = pyo.Constraint(self.model.t, rule=terminal_constraints_y)
        self.model.terminal_constraints_z = pyo.Constraint(self.model.t, rule=terminal_constraints_z)

        # Velocity
        self.model.terminal_constraints_vx = pyo.Constraint(self.model.t, rule=terminal_constraints_vx)
        self.model.terminal_constraints_vy = pyo.Constraint(self.model.t, rule=terminal_constraints_vy)
        self.model.terminal_constraints_vz = pyo.Constraint(self.model.t, rule=terminal_constraints_vz)

        # # Acceleration
        # self.model.terminal_constraints_ax = pyo.Constraint(self.model.t, rule=terminal_constraints_ax)
        # self.model.terminal_constraints_ay = pyo.Constraint(self.model.t, rule=terminal_constraints_ay)
        # self.model.terminal_constraints_az = pyo.Constraint(self.model.t, rule=terminal_constraints_az)
        #
        # # Angular acceleration
        # self.model.terminal_constraints_pdot = pyo.Constraint(self.model.t, rule=terminal_constraints_pdot)
        # self.model.terminal_constraints_qdot = pyo.Constraint(self.model.t, rule=terminal_constraints_qdot)
        # self.model.terminal_constraints_rdot = pyo.Constraint(self.model.t, rule=terminal_constraints_rdot)
        #
        # # Angular velocity
        # self.model.terminal_constraints_p = pyo.Constraint(self.model.t, rule=terminal_constraints_p)
        # self.model.terminal_constraints_q = pyo.Constraint(self.model.t, rule=terminal_constraints_q)
        # self.model.terminal_constraints_r = pyo.Constraint(self.model.t, rule=terminal_constraints_r)

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

    def discretize(self, nfe=5, ncp=3, scheme='LAGRANGE-RADAU'):
        print(f'Discretizing model with the following specs:')
        print(f'*Number of finite elements:\t{nfe}')
        print(f'*Number of collocation points:\t{ncp}')
        print(f'*Discretization scheme:\t{scheme}')
        discretizer = pyo.TransformationFactory('dae.collocation')
        discretizer.apply_to(self.instance, nfe=nfe, ncp=ncp, scheme=scheme)

    def solve(self, solver='ipopt', verbose=False):
        print(f'Solving DAE using {solver}')
        self.discretize()
        self.results = pyo.SolverFactory(solver).solve(self.instance, tee=verbose)
        self.to_historian()

    def create_model(self):
        print('Creating model')
        self.instance = self.model.create_instance()

    def print_results(self):
        # self.instance.display()
        self.instance.T1.pprint()
        # self.instance.terminal_constraints_vy.pprint()

    def to_historian(self):
        print('Writing results to historian')
        data = {'t': [t*pyo.value(self.instance.time) for t in list(self.instance.t.data())],
                'x': pyo.value(self.instance.x[:]),
                'y': pyo.value(self.instance.y[:]),
                'z': pyo.value(self.instance.z[:]),
                'roll': pyo.value(self.instance.roll[:]),
                'pitch': pyo.value(self.instance.pitch[:]),
                'yaw': pyo.value(self.instance.yaw[:]),
                'p': pyo.value(self.instance.p[:]),
                'q': pyo.value(self.instance.q[:]),
                'r': pyo.value(self.instance.r[:]),
                'T1': pyo.value(self.instance.T1[:]),
                'T2': pyo.value(self.instance.T2[:]),
                'T3': pyo.value(self.instance.T3[:]),
                'T4': pyo.value(self.instance.T4[:]),
                'vx': pyo.value(self.instance.vx[:]),
                'vy': pyo.value(self.instance.vy[:]),
                'vz': pyo.value(self.instance.vz[:]),
                'ax': pyo.value(self.instance.ax[:]),
                'ay': pyo.value(self.instance.ay[:]),
                'az': pyo.value(self.instance.az[:]),
                'pdot': pyo.value(self.instance.pdot[:]),
                'qdot': pyo.value(self.instance.qdot[:]),
                'rdot': pyo.value(self.instance.rdot[:])
                }
        self.historian = data

    def plot_property_from_historian(self, title, *args, save_folder=None, **kwargs):
        """
        Plots the list of dicts from the historian.
        :param title: string
        :param args: list of dicts
        :param save_folder: string
        :param kwargs:
        :return:
        """
        # If save_folder is not provided, use the default from self.config
        if save_folder is None:
            save_folder = self.config.get('output', '.')  # Fallback to '.' if 'output' key is missing
        print(f'Plotting {title}')
        fig, ax = plt.subplots(len(args))
        plt.title(f'{title}')
        for subplot_idx in range(len(args)):
            ax[subplot_idx].plot(self.historian[args[subplot_idx]['x_axis']],
                                 self.historian[args[subplot_idx]['y_axis']])
            ax[subplot_idx].set_ylabel(args[subplot_idx]['y_axis'])

        # Save the figure in the specified folder
        fig.savefig(os.path.join(save_folder, title), **kwargs)

    def plot_3d_property_from_historian(self, title, data, save_folder=None, **kwargs):
        # If save_folder is not provided, use the default from self.config
        if save_folder is None:
            save_folder = self.config.get('output', '.')  # Fallback to '.' if 'output' key is missing
        print(f'Plotting {title}')
        fig, ax = plt.subplots()
        ax3d = plt.axes(projection='3d')
        ax3d.scatter3D(self.historian[data['x_axis']],
                       self.historian[data['y_axis']],
                       self.historian[data['z_axis']],
                       alpha=[x/max(self.historian[data['transparency']])
                              for x in self.historian[data['transparency']]]
                       )
        plt.title(f'{title}')

        # Save the figure in the specified folder
        fig.savefig(os.path.join(save_folder, title), **kwargs)


    def plot_historian(self):
        print('Plot historian')
        self.plot_property_from_historian('Position',
                                          *[{'x_axis': 't', 'y_axis': 'x'},
                                            {'x_axis': 't', 'y_axis': 'y'},
                                            {'x_axis': 't', 'y_axis': 'z'}],
                                          dpi=300)

        self.plot_property_from_historian('Velocity',
                                          *[{'x_axis': 't', 'y_axis': 'vx'},
                                            {'x_axis': 't', 'y_axis': 'vy'},
                                            {'x_axis': 't', 'y_axis': 'vz'}],
                                          dpi=300
                                          )

        self.plot_property_from_historian('Acceleration',
                                          *[{'x_axis': 't', 'y_axis': 'ax'},
                                            {'x_axis': 't', 'y_axis': 'ay'},
                                            {'x_axis': 't', 'y_axis': 'az'}],
                                          dpi=300
                                          )

        self.plot_property_from_historian('Thrust',
                                          *[{'x_axis': 't', 'y_axis': 'T1'},
                                            {'x_axis': 't', 'y_axis': 'T2'},
                                            {'x_axis': 't', 'y_axis': 'T3'},
                                            {'x_axis': 't', 'y_axis': 'T4'}],
                                          dpi=300)

        self.plot_3d_property_from_historian('3D Projectile',
                                             {'x_axis': 'x', 'y_axis': 'y', 'z_axis': 'z', 'transparency': 't'},
                                             dpi=300)

    def set_initial_conditions(self):
        print('Setting initial conditions')

        def initial_condition_x(model, t):
            if t == 0:
                return model.x[t] == 0
            else:
                return pyo.Constraint.Skip

        def initial_condition_y(model, t):
            if t == 0:
                return model.y[t] == 0
            else:
                return pyo.Constraint.Skip

        def initial_condition_z(model, t):
            if t == 0:
                return model.z[t] == 0
            else:
                return pyo.Constraint.Skip

        def initial_condition_vx(model, t):
            if t == 0:
                return model.vx[t] == 1
            else:
                return pyo.Constraint.Skip

        def initial_condition_vy(model, t):
            if t == 0:
                return model.vy[t] == 1
            else:
                return pyo.Constraint.Skip

        def initial_condition_vz(model, t):
            if t == 0:
                return model.vz[t] == 1
            else:
                return pyo.Constraint.Skip

        def initial_condition_pdot(model, t):
            if t == 0:
                return model.pdot[t] == 0
            else:
                return pyo.Constraint.Skip

        def initial_condition_qdot(model, t):
            if t == 0:
                return model.qdot[t] == 0
            else:
                return pyo.Constraint.Skip

        def initial_condition_rdot(model, t):
            if t == 0:
                return model.rdot[t] == 0
            else:
                return pyo.Constraint.Skip

        def initial_condition_p(model, t):
            if t == 0:
                return model.p[t] == 0
            else:
                return pyo.Constraint.Skip

        def initial_condition_q(model, t):
            if t == 0:
                return model.q[t] == 0
            else:
                return pyo.Constraint.Skip

        def initial_condition_r(model, t):
            if t == 0:
                return model.r[t] == 0
            else:
                return pyo.Constraint.Skip

        # Position
        self.instance.initial_condition_x = pyo.Constraint(self.instance.t, rule=initial_condition_x)
        self.instance.initial_condition_y = pyo.Constraint(self.instance.t, rule=initial_condition_y)
        self.instance.initial_condition_z = pyo.Constraint(self.instance.t, rule=initial_condition_z)

        # Velocity
        self.instance.initial_condition_vx = pyo.Constraint(self.instance.t, rule=initial_condition_vx)
        self.instance.initial_condition_vy = pyo.Constraint(self.instance.t, rule=initial_condition_vy)
        self.instance.initial_condition_vz = pyo.Constraint(self.instance.t, rule=initial_condition_vz)

        # Angular acceleration
        self.instance.initial_condition_pdot = pyo.Constraint(self.instance.t, rule=initial_condition_pdot)
        self.instance.initial_condition_qdot = pyo.Constraint(self.instance.t, rule=initial_condition_qdot)
        self.instance.initial_condition_rdot = pyo.Constraint(self.instance.t, rule=initial_condition_rdot)

        # Angular velocity
        self.instance.initial_condition_p = pyo.Constraint(self.instance.t, rule=initial_condition_p)
        self.instance.initial_condition_q = pyo.Constraint(self.instance.t, rule=initial_condition_q)
        self.instance.initial_condition_r = pyo.Constraint(self.instance.t, rule=initial_condition_r)
