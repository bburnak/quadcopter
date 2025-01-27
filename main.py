import argparse
import yaml
import os
from functions.opt_model import DronePyomo
from functions.simulation import Drone
from functions.system_identification import SystemIdentification


def workflow_main(config):
    drone = Drone(config)
    drone.simulate()
    drone.plot()


def workflow_system_identification(config):
    print(f'Running system identification.')
    drone = Drone(config)
    si = SystemIdentification(drone)
    si.run()


def workflow_dynamic_optimization(config):
    print('Performing dynamic optimization')
    pyo_model = DronePyomo(config)
    pyo_model.create_model()
    pyo_model.set_initial_conditions()
    pyo_model.solve(verbose=True)
    pyo_model.plot_historian()
    pyo_model.print_results()


def parse_arguments() -> dict:
    """
    Reads the input arguments to the main code
    :return: dict of input arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='Absolute or relative input directory')
    parser.add_argument('-o', '--output', type=str, help='Absolute or relative output directory')
    args = parser.parse_args()
    return vars(args)


def get_default_config() -> dict:
    config = {
        'physics': {
            'gravitational_acceleration': 9.81
        },
        'specs': {
            'thrust': {
                'min': 0,
                'max': 10
            },
            'mass': 0.1,
            'arm_length': 0.25,
            'Ixx': 0.1,
            'Iyy': 0.1,
            'Izz': 0.1,
            'k_drag_linear': 0.1,
            'k_drag_angular': 0.05,
        },
        'system_defaults': {
            'delta_t': 0.1
        }
    }
    return config


def get_config_from_args(args: dict) -> dict:
    filepath = os.path.join(args['input'], 'config.yaml')

    try:
        with open(filepath) as stream:
            config = yaml.safe_load(stream)
    except FileNotFoundError:
        print('Failed to read config file. Using default configurations.')
        config = {}
    config['input'] = args['input']
    config['output'] = args['output']
    return config | get_default_config()


def run_cli(workflow):
    print(f'Running main script.')

    args = parse_arguments()
    config = get_config_from_args(args)

    if workflow == 'simulate':
        workflow_main(config)
    elif workflow == 'system_identification':
        workflow_system_identification()
    elif workflow == 'dynamic_optimization':
        workflow_dynamic_optimization(config)
    else:
        raise ValueError("Unknown workflow")


if __name__ == '__main__':
    run_cli('simulate')
