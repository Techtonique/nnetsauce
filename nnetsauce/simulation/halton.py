from sys import version_info as _swig_python_version_info

if _swig_python_version_info < (2, 7, 0):
    raise RuntimeError("Python 2.7 or later required")

    # Import the low-level C/C++ module
    # if __package__ or "." in __name__:

    try:

        from ._halton import *

    except:

        import _halton

    else:

        try:
            import builtins as __builtin__
        except ImportError:
            import __builtin__

        def _swig_repr(self):
            try:
                strthis = "proxy of " + self.this.__repr__()
            except __builtin__.Exception:
                strthis = ""
            return "<%s.%s; %s >" % (
                self.__class__.__module__,
                self.__class__.__name__,
                strthis,
            )

        def _swig_setattr_nondynamic_instance_variable(set):
            def set_instance_attr(self, name, value):
                if name == "thisown":
                    self.this.own(value)
                elif name == "this":
                    set(self, name, value)
                elif hasattr(self, name) and isinstance(
                    getattr(type(self), name), property
                ):
                    set(self, name, value)
                else:
                    raise AttributeError(
                        "You cannot add instance attributes to %s" % self
                    )

            return set_instance_attr

        def _swig_setattr_nondynamic_class_variable(set):
            def set_class_attr(cls, name, value):
                if hasattr(cls, name) and not isinstance(
                    getattr(cls, name), property
                ):
                    set(cls, name, value)
                else:
                    raise AttributeError(
                        "You cannot add class attributes to %s" % cls
                    )

            return set_class_attr

        def _swig_add_metaclass(metaclass):
            """Class decorator for adding a metaclass to a SWIG wrapped class - a slimmed down version of six.add_metaclass"""

            def wrapper(cls):
                return metaclass(
                    cls.__name__, cls.__bases__, cls.__dict__.copy()
                )

            return wrapper

        class _SwigNonDynamicMeta(type):
            """Meta class to enforce nondynamic attributes (no new attributes) for a class"""

            __setattr__ = _swig_setattr_nondynamic_class_variable(
                type.__setattr__
            )

        def halton(i, m):
            return _halton.halton(i, m)

        def halton_base(i, m, b):
            return _halton.halton_base(i, m, b)

        def halton_inverse(r, m):
            return _halton.halton_inverse(r, m)

        def halton_sequence(i1, i2, m):
            return _halton.halton_sequence(i1, i2, m)

        def i4vec_sum(n, a):
            return _halton.i4vec_sum(n, a)

        def prime(n):
            return _halton.prime(n)

        def r8_mod(x, y):
            return _halton.r8_mod(x, y)

        def timestamp():
            return _halton.timestamp()
