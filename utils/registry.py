from functools import wraps



class Registry:
    def __init__(self, name = None):
        self.name = name
        self._registry = {}

    def register(self, name, obj):
        if name in self._registry:
            raise ValueError(f'{name} is already registered')

        self._registry[name] = obj
    
    def __call__(self, name = None):
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