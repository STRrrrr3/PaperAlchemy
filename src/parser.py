from importlib import import_module as _import_module
import sys as _sys

_impl = _import_module('src.parsing.parser')
globals().update({name: getattr(_impl, name) for name in dir(_impl) if not name.startswith('__')})
_sys.modules[__name__] = _impl
