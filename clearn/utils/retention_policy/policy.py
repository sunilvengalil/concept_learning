import heapq
import traceback
from typing import List
import numpy as np

from itertools import count

tiebreaker = count()


from clearn.config import ExperimentConfig


class RetentionPolicy:
    def __init__(self,
                 data_type:str,
                 policy_type: str,
                 N: int
                 ):
        self.data_queue: List = []
        self.policy_type: str = policy_type
        self.N: int = N
        self.data_type = data_type

    def _update_heap(self, exp_config: ExperimentConfig,  costs: np.ndarray, data):

        """"
        Validation started
        """

        if len(data) != 4:
            raise ValueError(f"Parameter data should be a list of lenght 4. Got {type(data)} of length {len(data)} instead")

        reconstructed_images, labels, nlls, orig_images = data[0], data[1], data[2], data[3]
        if isinstance(costs, np.ndarray):
            costs = np.squeeze(costs)
            costs = costs.tolist()
        else:
            raise Exception(f" Data type of costs is {type(costs)}")

        # if len(costs) != exp_config.BATCH_SIZE:
        #     raise Exception(f"Shape of cost is {len(costs)} ")

        """
        Validation completed
        """

        if len(self.data_queue) == 0:
            current_max_in_heap = costs[0]
        else:
            current_max_in_heap = -self.data_queue[0][0]


        try:
            for cost, reconstructed_image, label, nll, orig_image in zip(costs, reconstructed_images, labels, nlls, orig_images):
                if len(self.data_queue) < self.N:
                    heapq.heappush(self.data_queue, (-cost,  next(tiebreaker), [reconstructed_image, label, nll, orig_image]))
                    if cost < current_max_in_heap:
                        current_max_in_heap = cost
                    # print("Cost", cost, current_max_in_heap)
                else:
                    if cost < current_max_in_heap:
                        _current_max_in_heap = heapq.heappushpop(self.data_queue, (-cost, next(tiebreaker),  [reconstructed_image, label, nll, orig_image]))
                        current_max_in_heap = -_current_max_in_heap[0]
        except:
            print(f" Type of cost {type(cost)}. Cost:{cost}")
            if isinstance(cost, list):
                print(f"Length of cost is {len(cost)}" )
            elif isinstance(cost, np.ndarray):
                print(f"Shape of cost is {cost.shape}")
            traceback.print_exc()

    def update_heap(self,  cost, exp_config,  data):
        if self.policy_type == "TOP":
            _cost = -cost
        elif self.policy_type == "BOTTOM":
            _cost = cost
        else:
            raise Exception(f"Allowed policy_types are TOP and BOTTOM. Got {self.policy_type} instead")
        if len(self.data_queue) == 0:
            self._update_heap(exp_config, _cost, data)
        else:
            if min(cost) < -self.data_queue[0][0]:
                self._update_heap(exp_config,  _cost, data)
