from functools import wraps



class Registry:
    '''A simple registry for mapping names to objects.'''

    def __init__(self, name: str | None = None) -> None:
        '''Initialize the registry.

        Args:
            name (str | None): Optional name for the registry.
        '''
        self.name = name
        self._registry = {}

    def register(self, name: str, obj: type) -> None:
        '''Register an object under the given name.

        Args:
            name (str): The name to register the object under.
            obj (type): The object to register.

        Raises:
            ValueError: If the name is already registered.
        '''
        if name in self._registry:
            raise ValueError(f'{name} is already registered')

        self._registry[name] = obj
    
    def __call__(self, name: str | None = None):
        '''Decorator to register a class or function.

        Args:
            name (str | None): The name to register under. If None, uses the object's __name__.

        Returns:
            callable: A decorator that registers the object and returns it.
        '''
        def _register(obj):
            self.register(name or obj.__name__, obj)
            return obj
        return _register

    def __getitem__(self, key):
        return self._registry[key]
    
    def __contains__(self, key):
        return key in self._registry
    
    def keys(self):
        return self._registry.keys()
    
    def items(self):
        return self._registry.items()

SCALER_REGISTRY = Registry("scaler")
DATASET_REGISTRY = Registry("dataset")
MODEL_REGISTRY = Registry("model")
OPTIMIZER_REGISTRY = Registry("optimizer")
TRAINER_REGISTRY = Registry("trainer")
EVALUATOR_REGISTRY = Registry("evaluator")