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
        5: (2, 3)
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


def add_to_table(table, new_table, row_ix):
    i = 0
    for new in new_table:
        table[row_ix][i] = new
        i = i + 1

    return table


def calculate_mean_std(table_peaks, table_heights, peaks, precision_rate):
    max_peaks = len(peaks)
    repetitions = len(table_peaks)
    period = len(table_peaks[0])

    values_peaks = np.empty((max_peaks, repetitions))
    values_peaks[:] = np.nan
    mean_N_std_p = np.empty((3, (max_peaks)))
    mean_N_std_p[:] = np.nan

    values_heights = np.empty((max_peaks, repetitions))
    values_heights[:] = np.nan
    mean_N_std_h = np.empty((3, (max_peaks)))
    mean_N_std_h[:] = np.nan

    ix = 0
    for peak in peaks:
        N = 0
        mean_p = 0
        mean_h = 0
        k = 0
        for i in range(period):
            for j in range(repetitions):
                value_p = table_peaks[j][i]
                value_h = table_heights[j][i]
                if not np.isnan(value_p):
                    if math.isclose(value_p, peak, abs_tol=precision_rate):
                        N = N + 1
                        mean_p = mean_p + value_p
                        mean_h = mean_h + value_h
                        values_peaks[ix][k] = value_p
                        values_heights[ix][k] = value_h
                        k = k + 1
                        if k == repetitions:
                            break
            if k == repetitions:
                break

        if N != 0:
            mean_N_std_p[0][ix] = mean_p / N
            mean_N_std_p[1][ix] = N
            mean_N_std_h[0][ix] = mean_h / N
            mean_N_std_h[1][ix] = N

            ix = ix + 1

    ix = 0
    for i in range(max_peaks):
        sum_p = 0
        sum_h = 0
        if not np.isnan(mean_N_std_p[0][i]):
            for j in range(repetitions):
                if not np.isnan(values_peaks[i][j]):
                    sum_p = sum_p + (values_peaks[i][j] - mean_N_std_p[0][i]) ** 2
                    sum_h = sum_h + (values_heights[i][j] - mean_N_std_h[0][i]) ** 2
            mean_N_std_p[2][i] = math.sqrt((1 / mean_N_std_p[1][i]) * sum_p)
            mean_N_std_h[2][i] = math.sqrt((1 / mean_N_std_h[1][i]) * sum_h)

            if ix == 0:
                mean_std_p = np.array(np.array([mean_N_std_p[0][i], mean_N_std_p[2][i]]))
                mean_std_h = np.array(np.array([mean_N_std_h[0][i], mean_N_std_h[2][i]]))
            else:
                mean_std_p = np.row_stack((mean_std_p, np.array([mean_N_std_p[0][i], mean_N_std_p[2][i]])))
                mean_std_h = np.row_stack((mean_std_h, np.array([mean_N_std_h[0][i], mean_N_std_h[2][i]])))
            ix = ix + 1

    if ix == 0:
        return [],[]
    else:
        return mean_std_p, mean_std_h
