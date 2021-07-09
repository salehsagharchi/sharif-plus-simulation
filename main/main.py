import os
import pickle
from datetime import datetime

import simulation_system

random_filename = "RandomObjects/2021_07_09_03_10_0xx8.obj"


def load_random_gen():
    if random_filename == "" or not os.path.exists(random_filename):
        return simulation_system.RandomGen()
    else:
        with open(random_filename, 'rb') as f:
            return pickle.load(f)


def save_random_gen(to_save):
    with open(datetime.today().strftime('RandomObjects/%Y_%m_%d_%H_%M_%S.obj'), 'wb') as f:
        pickle.dump(to_save, f)


def start_simulation():
    random_gen = load_random_gen()
    try:
        simulation_sytem = simulation_system.Simulation(1000000, random_gen)
        simulation_sytem.initialize()
        simulation_sytem.start_simulation()
        print("Simulation Finished")
    finally:
        if random_gen:
            random_gen.old_random_list = random_gen.new_random_list
            random_gen.new_random_list = []
            save_random_gen(random_gen)


if __name__ == '__main__':
    start_simulation()
