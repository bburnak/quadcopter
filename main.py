from model import Drone


def run_cli():
    print(f'Running main script.')
    drone = Drone()
    drone.simulate()
    drone.plot()


if __name__ == '__main__':
    run_cli()

