import heapq
from typing import List


class RetentionPolicy:
    def __init__(self, policy_type: str, N: int):
        self.data_queue: List = []
        self.policy_type: str = policy_type
        self.N: int = N

    def _update_heap(self, costs, data):
        if len(self.data_queue) == 0:
            current_max_in_heap = costs[0]
        else:
            current_max_in_heap = -self.data_queue[0][0]
        reconstructed_images, labels, nlls = data[0], data[1], data[2]
        for cost, reconstructed_image, label, nll in zip(costs, reconstructed_images, labels, nlls):
            if len(self.data_queue) < self.N:
                heapq.heappush(self.data_queue, (-cost, [reconstructed_image, label, nll]))
                if cost < current_max_in_heap:
                    current_max_in_heap = cost
            else:
                if cost < current_max_in_heap:
                    _current_max_in_heap = heapq.heappushpop(self.data_queue, (-cost, [reconstructed_image, label, nll]))
                    current_max_in_heap = -_current_max_in_heap[0]

    def update_heap(self, cost, data):
        if self.policy_type == "TOP":
            _cost = -cost
        elif self.policy_type == "BOTTOM":
            _cost = cost
        else:
            raise Exception(f"Allowed policy_types are TOP and BOTTOM. Got {self.policy_type} instead")
        if len(self.data_queue) == 0:
            self._update_heap(_cost, data)
        else:
            if min(cost) < -self.data_queue[0][0]:
                self._update_heap(_cost, data)
