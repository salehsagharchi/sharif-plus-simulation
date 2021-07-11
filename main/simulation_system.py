import enum
from typing import List, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


class CustomerPosition(enum.Enum):
    NotArrived = -1
    InQueue = 1
    Serving = 2
    Canceled = 3
    Exited = 4


class Customer:
    def __init__(self, cid, alpha_rate: float, parts_count, random_gen):
        self.random_gen = random_gen
        self.cid = cid
        priority_selector = self.random_gen.get_randint(0, 100)
        self.priority = 0
        if 50 < priority_selector < 70:
            self.priority = 1
        elif 70 < priority_selector < 85:
            self.priority = 2
        elif 85 < priority_selector < 95:
            self.priority = 3
        elif 95 < priority_selector < 100:
            self.priority = 4
        self.patience_period = self.random_gen.get_random_period(alpha_rate)
        self.patience_time = -1
        self.request_part = int(self.random_gen.get_randint(1, 1 + parts_count))

        self.arrival_time = np.inf
        self.reception_start_time = np.inf
        self.reception_end_time = np.inf
        self.cook_start_time = np.inf
        self.cook_end_time = np.inf

        self.position: Tuple[int, CustomerPosition] = (-1, CustomerPosition.NotArrived)  # (Queue Number, State)


class Server:
    def __init__(self, is_reception, max_priority: int, worker_rates: List[float], simulation_system):
        self.simulation_system: Simulation = simulation_system
        self.random_gen = self.simulation_system.random_gen
        self.is_reception = is_reception
        self.queues: Dict[int, List[Customer]] = {}
        self.max_priority = max_priority
        for i in range(max_priority + 1):
            self.queues[i] = []
        self.worker_count = len(worker_rates)
        self.worker_rates = worker_rates
        self.worker_working_on: List[Optional[Customer]] = [None for _ in range(self.worker_count)]
        self.worker_next_event_time: List[Tuple[int, int]] = [(np.inf, np.inf) for _ in range(
            self.worker_count)]  # (cooking end time, patience time)

    def enter_customer(self, new_customer: Customer):
        self.queues[new_customer.priority].append(new_customer)
        self.start_serving()

    def next_customer(self) -> Optional[Customer]:
        for i in range(self.max_priority, -1, -1):
            while len(self.queues[i]) > 0:
                first_item = self.queues[i].pop(0)
                if first_item.position[1] != CustomerPosition.Canceled and first_item.patience_time > self.get_time():
                    return first_item
                else:
                    self.simulation_system.tired_in_queues[first_item.position[0]] -= 1
        return None

    def start_serving(self):
        idles = []
        for i, working_on in enumerate(self.worker_working_on):
            if working_on is None:
                idles.append(i)

        while len(idles) > 0:
            customer = self.next_customer()
            if customer is None:
                return
            chosen_worker = idles[self.random_gen.get_randint(0, len(idles))]
            idles.remove(chosen_worker)
            self.worker_working_on[chosen_worker] = customer

            if self.is_reception:
                customer.reception_start_time = self.get_time()
            else:
                customer.cook_start_time = self.get_time()
            customer.position = (customer.position[0], CustomerPosition.Serving)

            serving_period = self.random_gen.get_random_period(self.worker_rates[chosen_worker])
            self.worker_next_event_time[chosen_worker] = (self.get_time() + serving_period, customer.patience_time)

    def time_changed(self):
        now_time = self.get_time()
        for i in range(self.worker_count):
            if self.worker_next_event_time[i][0] == now_time:
                customer = self.worker_working_on[i]
                assert customer is not None
                assert customer.position[1] == CustomerPosition.Serving
                self.simulation_system.customer_work_finished(customer)
                self.worker_working_on[i] = None
                self.worker_next_event_time[i] = (np.inf, np.inf)
            elif self.worker_next_event_time[i][1] == now_time:
                customer = self.worker_working_on[i]
                self.simulation_system.tired_in_queues[customer.position[0]] -= 1
                self.worker_working_on[i] = None
                self.worker_next_event_time[i] = (np.inf, np.inf)
        self.start_serving()

    def get_queue_length(self):
        length = 0
        for queue in self.queues.values():
            length += len(queue)
        return length

    def nearest_event(self):
        return min(self.worker_next_event_time, key=lambda x: x[0])[0]

    def is_idle(self):
        for queue in self.queues.values():
            if len(queue) > 0:
                return False
        for working_on in self.worker_working_on:
            if working_on is not None:
                return False
        return True

    def get_time(self):
        return self.simulation_system.now_time


class Simulation:
    def __init__(self, population, random_gen):
        self.random_gen = random_gen
        self.population = population
        self.parts_count = 0
        self.arrival_rate = 0
        self.reception_rate = 0
        self.alpha_rate = 0
        self.cook_rates: List[List[float]] = []

        self.servers: List[Server] = []

        self.last_cid = -1
        self.entered_customers = 0
        self.canceled_customers = 0
        self.exited_customers = 0

        self.now_time = 0

        self.next_arrival_time = np.inf

        self.all_customers: Dict[int, Customer] = {}
        self.patience_times: List[Tuple[int, int]] = []  # [(time, id), ...]

        self.total_system_times: np.ndarray = np.empty((self.population, 2), dtype=np.int64)
        self.total_queue_times: np.ndarray = np.empty((self.population, 2), dtype=np.int64)
        self.queue_length_in_times: List[List[Tuple[int, int]]] = []  # [ [(time, length), ...] , ... ]
        self.queue_length_in_periods: List[List[Tuple[int, int]]] = []  # [ [(period, length), ...] , ... ]
        self.tired_in_queues: np.ndarray = np.array([])

        self.presents_in_system: List[Tuple[int, int]] = []  # [(time, number), ...]

    def initialize(self):
        line = input("ENTER 𝑁, 𝜆, 𝜇, a : ")
        self.parts_count = int(line.split(",")[0].strip())
        self.arrival_rate = float(line.split(",")[1].strip())
        self.reception_rate = float(line.split(",")[2].strip())
        self.alpha_rate = float(line.split(",")[3].strip())
        for i in range(self.parts_count):
            line = input(f"ENTER cook rates for part {i + 1} : ")
            self.cook_rates.append(list(map(lambda x: float(x.strip()), line.split(","))))

        for i in range(self.parts_count + 1):
            if i == 0:
                server_item = Server(True, 4, [self.reception_rate], self)
            else:
                server_item = Server(False, 4, self.cook_rates[i - 1], self)
            self.servers.append(server_item)

        for i in range(self.parts_count + 1):
            self.queue_length_in_times.append([])
            self.queue_length_in_periods.append([])
        self.tired_in_queues = np.empty(len(self.servers), dtype=np.int64)
        self.tired_in_queues.fill(0)

    def customer_work_finished(self, customer: Customer):
        if customer.position[0] == 0:
            customer.reception_end_time = self.now_time
            customer.position = (customer.request_part, CustomerPosition.InQueue)
            self.servers[customer.request_part].enter_customer(customer)
        else:
            customer.cook_end_time = self.now_time
            customer.position = (customer.position[0], CustomerPosition.Exited)
            self.exited_customers += 1

    def is_simulation_finished(self):
        if self.entered_customers < self.population:
            return False
        for server in self.servers:
            if not server.is_idle():
                return False
        return True

    def handle_tiring(self):
        while len(self.patience_times) > 0:
            to_tire = self.patience_times[0]
            if to_tire[0] > self.now_time:
                return
            customer = self.all_customers[to_tire[1]]
            if customer.position[1] == CustomerPosition.Exited:
                self.patience_times.pop(0)
                continue
            customer.position = (customer.position[0], CustomerPosition.Canceled)
            self.canceled_customers += 1
            self.tired_in_queues[customer.position[0]] += 1
            self.patience_times.pop(0)

    def start_simulation(self):
        last_time = 0
        print(f'Simulation Start for population {self.population}')
        while not self.is_simulation_finished():
            assert self.now_time != np.inf
            if self.now_time == self.next_arrival_time:
                self.next_arrival_time = np.inf
                self.last_cid += 1
                new_customer = Customer(self.last_cid, self.alpha_rate, self.parts_count, self.random_gen)
                new_customer.position = (0, CustomerPosition.InQueue)
                new_customer.arrival_time = self.now_time
                new_customer.patience_time = new_customer.patience_period + self.now_time
                self.servers[0].enter_customer(new_customer)
                self.entered_customers += 1
                self.all_customers[new_customer.cid] = new_customer
                self.patience_times.append((new_customer.patience_time, new_customer.cid))
                self.patience_times.sort(key=lambda item: item[0])

                # print(f'new customer {self.last_cid} entered { self.entered_customers < self.population}')

            if self.next_arrival_time == np.inf and self.entered_customers < self.population:
                self.next_arrival_time = self.now_time + self.random_gen.get_random_period(self.arrival_rate)

            for server in self.servers:
                server.time_changed()

            self.handle_tiring()

            for i, server in enumerate(self.servers):
                self.queue_length_in_times[i].append(
                    (self.now_time, server.get_queue_length() - self.tired_in_queues[i]))

            self.presents_in_system.append(
                (self.now_time, self.entered_customers - self.exited_customers - self.canceled_customers))

            last_time = self.now_time

            next_time = self.next_arrival_time
            if len(self.patience_times) > 0:
                next_time = min(next_time, self.patience_times[0][0])
            for server in self.servers:
                next_time = min(next_time, server.nearest_event())

            self.now_time = next_time

        self.now_time = last_time

    def analyze_result(self):
        assert self.entered_customers == self.exited_customers + self.canceled_customers
        assert self.population == len(self.all_customers)

        for i in self.all_customers:
            customer = self.all_customers[i]
            if customer.position[1] == CustomerPosition.Exited:
                system_time = customer.cook_end_time - customer.arrival_time
                queue_time = (customer.reception_start_time - customer.arrival_time) + (customer.cook_start_time
                                                                                        - customer.reception_end_time)
            elif customer.position[1] == CustomerPosition.Canceled:
                system_time = customer.patience_time - customer.arrival_time
                if customer.reception_start_time == np.inf:
                    queue_time = system_time
                elif customer.reception_end_time == np.inf:
                    queue_time = customer.reception_start_time - customer.arrival_time
                elif customer.cook_start_time == np.inf:
                    queue_time = system_time - (customer.reception_end_time - customer.reception_start_time)
                else:
                    queue_time = (customer.reception_start_time - customer.arrival_time) + (customer.cook_start_time -
                                                                                            customer.reception_end_time)
            else:
                raise Exception("Cannot analyze result before ending simulation!")
            self.total_system_times[i][0] = system_time
            self.total_system_times[i][1] = customer.priority
            self.total_queue_times[i][0] = queue_time
            self.total_queue_times[i][1] = customer.priority

        for i in range(len(self.servers)):
            for j in range(len(self.queue_length_in_times[i]) - 1):
                item = self.queue_length_in_times[i][j]
                item2 = self.queue_length_in_times[i][j + 1]
                self.queue_length_in_periods[i].append((item2[0] - item[0], item[1]))

        print(f"\nTotal time spent in seconds = {self.now_time}")
        print("#" * 30)
        priority_filters = [np.array([i]) for i in range(0, 5)]
        print(f"Average time spent in system for TOTAL priority = {np.mean(self.total_system_times[:, 0])}")
        for i in range(0, 5):
            to_mean = self.total_system_times[np.in1d(self.total_system_times[:, 1], priority_filters[i])][:, 0]
            if len(to_mean) > 0:
                print(f"Average time spent in system for priority {i} = {np.mean(to_mean)}")

        print("#" * 30)
        print(f"Average time spent in queue for TOTAL priority = {np.mean(self.total_queue_times[:, 0])}")
        for i in range(0, 5):
            to_mean = self.total_queue_times[np.in1d(self.total_queue_times[:, 1], priority_filters[i])][:, 0]
            if len(to_mean) > 0:
                print(f"Average time spent in queue for priority {i} = {np.mean(to_mean)}")

        print("#" * 30)
        print(f"Number of canceled customers = {self.canceled_customers}")

        print("#" * 30)
        for i in range(len(self.servers)):
            length = np.array(self.queue_length_in_periods[i])
            name = "Reception" if i == 0 else f"Part {i}"
            print(f"Average queue length in {name} queue = {np.average(length[:, 1], weights=length[:, 0])}")

        self.draw_plots()

    def draw_plots(self):
        rows = max(5, len(self.servers) + 1)
        figure, axis = plt.subplots(rows, 2, figsize=(20, 10 * rows))
        step = int(self.now_time / 100)
        for i in range(len(self.servers)):
            name = "Reception" if i == 0 else f"Part {i}"
            length = np.array(self.queue_length_in_times[i])
            periods = np.array(self.queue_length_in_periods[i])
            X = []
            Y = []
            last_ploted = 0
            for j in range(len(length)):
                if j >= len(periods):
                    break
                if length[j][0] - length[last_ploted][0] >= step:
                    X.append(length[last_ploted][0])
                    Y.append(np.average(periods[last_ploted:j, 1], weights=periods[last_ploted:j, 0]))
                    last_ploted = j
            axis[i, 0].plot(X, Y)
            axis[i, 0].set_title(f"Queue length for -> {name}")

        index = len(self.servers)
        present = np.array(self.presents_in_system)
        X = []
        Y = []
        last_ploted = 0
        for j in range(len(present)):
            if present[j][0] - present[last_ploted][0] >= step:
                X.append(present[last_ploted][0])
                Y.append(np.sum(present[last_ploted:j, 1]) / (present[j][0] - present[last_ploted][0]))
                last_ploted = j

        axis[index, 0].plot(X, Y)
        axis[index, 0].set_title(f"Number of presents in system")

        priority_filters = [np.array([i]) for i in range(0, 5)]
        for i in range(0, 5):
            to_mean = self.total_queue_times[np.in1d(self.total_queue_times[:, 1], priority_filters[i])][:, 0]
            if len(to_mean) > 0:
                axis[i, 1].hist(to_mean, rwidth=0.8)
                axis[i, 1].set_title(f"Histogram for waiting time for priority {i}")

        plt.show()


class RandomGen:
    def __init__(self):
        self.old_random_list = []
        self.new_random_list = []

    def get_random_period(self, rate):
        if len(self.old_random_list) > 0:
            rnd = self.old_random_list.pop(0)
            self.new_random_list.append(rnd)
            return rnd
        rnd = int(np.random.exponential(1.0 / rate) + 0.5)
        if rnd == 0:
            rnd = 1
        self.new_random_list.append(rnd)
        return rnd

    def get_randint(self, low, high):
        if len(self.old_random_list) > 0:
            rnd = self.old_random_list.pop(0)
            self.new_random_list.append(rnd)
            assert low <= rnd < high
            return rnd
        rnd = np.random.randint(low, high)
        self.new_random_list.append(rnd)
        return rnd
