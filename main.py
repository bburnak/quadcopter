import argparse
import yaml
import os
from functions.model import Drone, DronePyomo
from functions.system_identification import SystemIdentification


def workflow_main():
    drone = Drone()
    drone.simulate()
    drone.plot()


def workflow_system_identification():
    print(f'Running system identification.')
    drone = Drone()
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
    return {}


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
        workflow_main()
    elif workflow == 'system_identification':
        workflow_system_identification()
    elif workflow == 'dynamic_optimization':
        workflow_dynamic_optimization(config)
    else:
        raise ValueError("Unknown workflow")


if __name__ == '__main__':
    run_cli('dynamic_optimization')
