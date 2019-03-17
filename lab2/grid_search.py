import itertools
from random import shuffle
from tqdm import tqdm
import time


class GridSearch:
    def __init__(self, hp, fit, metric, rnd_count=None):
        self.hp = hp
        self.models = []
        self.fit = fit
        self.rnd_count = rnd_count
        self.results = []
        self.metric = metric

        self.cartesian_product_hp()

    def cartesian_product_hp(self):
        hp_names = self.hp.keys()

        for hp in itertools.product(*self.hp.values()):
            model = {}

            for n, hp_v in zip(hp_names, hp):
                model[n] = hp_v

            self.models.append(model)

    def execute(self):
        models = self.models

        if self.rnd_count is not None:
            models = shuffle(models)
            models = models[:self.rnd_count]

        for m in tqdm(models):
            start_time = time.perf_counter()
            result = self.fit(m)
            elapsed_time = time.perf_counter() - start_time

            self.results.append({
                'model': m,
                'result': result,
                'time': elapsed_time
            })

    def get_total_time(self):
        return sum(list(map(lambda t: t['time'], self.results)))

    def get_best_result(self, desc=True):
        return sorted(list(map(lambda t: t[self.metric], self.results)), reverse=desc)
