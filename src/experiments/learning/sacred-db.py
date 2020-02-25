from sacred import Experiment
from sacred.observers import MongoObserver

ex = Experiment('my_experiment')

ex.observers.append(MongoObserver.create())