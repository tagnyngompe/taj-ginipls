from sacred import Experiment
from sacred.observers import MongoObserver

ex = Experiment('hello_config')
ex.observers.append(MongoObserver.create())

@ex.config
def my_config():
    recipient = "tagny"
    message = "Hello %s!" % recipient

@ex.automain
def my_main(message):
    print(message)