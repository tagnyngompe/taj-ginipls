'''
$(py37)> python sacred-metrics.py -m sacred -F ../../../reports/sacred-metrics2
'''

import time
from numpy import random
from sacred import Experiment

ex = Experiment('example_metrics')


@ex.automain
def example_metrics(_run):
    counter = 0
    while counter < 20:
        counter += 1
        print("counter: ", counter)
        value = counter
        ms_to_wait = random.randint(5, 5000)
        time.sleep(ms_to_wait / 1000)
        # This will add an entry for training.loss metric in every second iteration.
        # The resulting sequence of steps for training.loss will be 0, 2, 4, ...
        if counter % 2 == 0:
            _run.log_scalar("training.loss", value * 1.5, counter)
        # Implicit step counter (0, 1, 2, 3, ...)
        # incremented with each call for training.accuracy:
        _run.log_scalar("training.accuracy", value * 2)
        # Another option is to use the Experiment object (must be running)
        # The training.diff has its own step counter (0, 1, 2, ...) too
        ex.log_scalar("training.diff", value * 2)
