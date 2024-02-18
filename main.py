from functions.model import Drone
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


def run_cli(workflow):
    if workflow == 'main':
        workflow_main()
    elif workflow == 'system_identification':
        workflow_system_identification()


if __name__ == '__main__':
    run_cli('system_identification')

