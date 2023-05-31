from abc import ABC
from torch.optim import SGD


class VROptimizer(ABC):
    def __init__(self, name, model, train_set, batch_size, lr, device, num_workers, weight_decay, noise):
        self.name = name
        self.model = model
        self.train_set = train_set
        self.batch_size = batch_size
        self.optimiser = SGD(model.parameters(), lr)
        self.device = device
        self.num_workers = num_workers
        self.noise = noise

    def run_one_iter(self):
        raise NotImplementedError
