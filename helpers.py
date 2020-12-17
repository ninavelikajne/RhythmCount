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


def add_peaks(peaks, new_peaks, row_ix):
    for peak in new_peaks:
        ix = int(round(peak))
        peaks[row_ix][ix] = peak

    return peaks


def add_heights(heights, new_heights, peaks, row_ix):
    i = 0
    for height in new_heights:
        ix = int(round(peaks[i]))
        heights[row_ix][ix] = height
        i = i + 1

    return heights


def calculate_mean_std(table):
    repetitions = len(table)
    period = len(table[0])
    mean_N_std = np.empty((3, period))
    mean_N_std[:] = np.nan
    for i in range(period):
        N = 0
        mean = 0
        for j in range(repetitions):
            value = table[j][i]
            if not np.isnan(value):
                N = N + 1
                mean = mean + value
        if N != 0:
            mean_N_std[0][i] = mean / N
            mean_N_std[1][i] = N

    for i in range(period):
        if mean_N_std[1][i] < repetitions * 0.50:
            mean_N_std[0][i] = np.nan
            mean_N_std[1][i] = np.nan

    ix = 0
    for i in range(period):
        sum = 0
        if not np.isnan(mean_N_std[0][i]):
            for j in range(repetitions):
                if not np.isnan(table[j][i]):
                    sum = sum + (table[j][i] - mean_N_std[0][i]) ** 2
            mean_N_std[2][i] = math.sqrt((1 / mean_N_std[1][i]) * sum)
            temp = np.array([mean_N_std[0][i], mean_N_std[2][i]])
            if ix == 0:
                mean_std = np.array(temp)
            else:
                mean_std = np.row_stack((mean_std, temp))
            ix = ix + 1

    if ix==0:
        return []
    else:
        return mean_std
