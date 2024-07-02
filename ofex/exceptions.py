import inspect


class OfexTypeError(TypeError):
    def __init__(self, *obj):
        typs = ' and \n'.join([f"{type(o)}({inspect.getmodule(type(o)).__file__})" for o in obj])
        super(OfexTypeError, self).__init__(
            typs + f" is not a valid type."
        )

