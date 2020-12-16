import numpy as np
import math
import os


class lazy_cartesian_product:
    def __init__(self, sets):
        self.sets = sets
        self.divs = []
        self.mods = []
        self.max_size = 1
        self.precompute()

    def precompute(self):
        for i in self.sets:
            self.max_size = self.max_size * len(i)
        length = len(self.sets)
        factor = 1
        for i in range((length - 1), -1, -1):
            items = len(self.sets[i])
            self.divs.insert(0, factor)
            self.mods.insert(0, items)
            factor = factor * items

    def entry_at(self, n):
        length = len(self.sets)
        if n < 0 or n >= self.max_size:
            raise IndexError
        combination = []
        for i in range(0, length):
            combination.append(self.sets[i][int(math.floor(n / self.divs[i])) % self.mods[i]])
        return combination


def criterium_value(criterium):
    values = dict({
        "BIC": True,
        "AIC": True,
        "log_likelihood": False
    })

    return values[criterium]


def get_factors(n):
    grid = dict({
        1: (1, 1),
        2: (2, 1),
        3: (2, 2),
        4: (2, 2),
        5: (3, 2)
    })

    return grid[n]


def make_results_dir():
    access_rights = 0o755
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, r'results')
    if not os.path.exists(final_directory):
        os.makedirs(final_directory, access_rights)


def get_model_name(model_type):
    names = dict({
        "poisson": "Poisson Model",
        "nb": "Negative Binomial Model",
        "gen_poisson": "Generalized Poisson Model",
        "zero_nb": "Zero-Inflated Negative Binomial Model",
        "zero_poisson": "Zero-Inflated Poisson Model"
    })
    return names[model_type]


def phase_to_radians(phase, period=24):
    return -(phase / period) * 2 * np.pi


