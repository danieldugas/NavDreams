import numpy as np
import torch

class Normalizer(object):


    def __init__(self, size, eps=0.02, clip=np.inf):
        self.eps = eps
        self.clip = clip
        self.mean = np.zeros(size)
        self.mean_sq = np.zeros(size)
        self.std = np.ones(size)
        self.count = 0

        self.new_count = 0
        self.new_sum = np.zeros_like(self.mean)
        self.new_sum_sq = np.zeros_like(self.mean_sq)
        return

    def record(self, x):
        size = self.get_size()
        is_array = isinstance(x, np.ndarray)
        if not is_array:
            assert (size == 1)
            x = np.array([[x]])

        x = np.reshape(x, [-1, size])

        self.new_count += x.shape[0]
        self.new_sum += np.sum(x, axis=0)
        self.new_sum_sq += np.sum(np.square(x), axis=0)
        return

    def update(self):
        new_count = self.new_count
        new_sum = self.new_sum
        new_sum_sq = self.new_sum_sq

        new_total = self.count + new_count

        if new_count > 0:
            new_mean = new_sum / new_count
            new_mean_sq = new_sum_sq / new_count

            w_old = float(self.count) / new_total
            w_new = float(new_count) / new_total

            self.mean = w_old * self.mean + w_new * new_mean
            self.mean_sq = w_old * self.mean_sq + w_new * new_mean_sq
            self.count = new_total
            self.std = self.calc_std(self.mean, self.mean_sq)

            self.new_count = 0
            self.new_sum.fill(0)
            self.new_sum_sq.fill(0)

        return

    def get_size(self):
        return self.mean.size

    def set_mean_std(self, mean, std):
        size = self.get_size()
        is_array = isinstance(mean, np.ndarray) and isinstance(std, np.ndarray)

        if not is_array:
            assert (size == 1)
            mean = np.array([mean])
            std = np.array([std])

        self.mean = mean
        self.std = std
        self.mean_sq = self.calc_mean_sq(self.mean, self.std)
        return

    def normalize(self, x):
        mean = torch.Tensor(self.mean)
        std = torch.Tensor(self.std)
        norm_x = (x - mean) / std
        #norm_x = np.clip(norm_x, -self.clip, self.clip)
        return norm_x

    def unnormalize(self, norm_x):
        x = norm_x * self.std + self.mean
        return x

    def calc_std(self, mean, mean_sq):
        var = mean_sq - np.square(mean)
        # some time floating point errors can lead to small negative numbers
        var = np.maximum(var, 0)
        std = np.sqrt(var)
        std = np.maximum(std, self.eps)
        return std

    def calc_mean_sq(self, mean, std):
        return np.square(std) + np.square(self.mean)

