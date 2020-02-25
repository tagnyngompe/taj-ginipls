from sacred import Experiment
from sacred.observers import MongoObserver

ex = Experiment('config_demo')
ex.observers.append(MongoObserver.create())


@ex.config
def my_config():
    """This is my demo configuration"""

    a = 10  # some integer

    # a dictionary
    foo = {
        'a_squared': a**2,
        'bar': 'my_string%d' % a
    }
    if a > 8:
        # cool: a dynamic entry
        e = a/2

@ex.named_config
def variant1():
    a = 100
    c = "bar"


ex.add_config({
  'foo': 42,
  'bar': 'baz'
})

@ex.command
def scream():
    """
    scream, and shout, and let it all out ...
    """
    print('AAAaaaaaaaahhhhhh...')

# ...

@ex.command
def greet(name):
    """
    Print a simple greet message.
    """
    print('Hello %s!' % name)

@ex.command(unobserved=True)
def helper(name):
    print('Running this command will not result in a DB entry!')
    greet(name)

@ex.automain
def run():
    pass
