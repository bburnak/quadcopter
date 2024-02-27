from functions.model import Drone, DronePyomo
from functions.system_identification import SystemIdentification


def workflow_main():
    print(f'Running main script.')
    drone = Drone()
    drone.simulate()
    drone.plot()


def workflow_system_identification():
    print(f'Running system identification.')
    drone = Drone()
    si = SystemIdentification(drone)
    si.run()


def workflow_dynamic_optimization():
    print('Performing dynamic optimization')
    pyo_model = DronePyomo()
    pyo_model.create_model()
    pyo_model.discretize()
    pyo_model.solve(verbose=True)



def run_cli(workflow):
    if workflow == 'main':
        workflow_main()
    elif workflow == 'system_identification':
        workflow_system_identification()
    elif workflow == 'dynamic_optimization':
        workflow_dynamic_optimization()
    else:
        raise ValueError("Unknown workflow")


if __name__ == '__main__':
    run_cli('dynamic_optimization')

