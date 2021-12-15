# adapted from sklearn

import sys
import os

from nnetsauce._build_utils import cythonize_extensions


def configuration(parent_package="", top_path=None):
    from numpy.distutils.misc_util import Configuration
    import numpy

    libraries = []
    if os.name == "posix":
        libraries.append("m")

    config = Configuration("nnetsauce", parent_package, top_path)

    # submodules with build utilities
    config.add_subpackage("__check_build")
    config.add_subpackage("_build_utils")

    # submodules which do not have their own setup.py
    # we must manually add sub-submodules & tests
    config.add_subpackage("demo")
    config.add_subpackage("utils")
    config.add_subpackage("base")
    config.add_subpackage("boosting")
    config.add_subpackage("custom")
    config.add_subpackage("glm")
    config.add_subpackage("mts")
    config.add_subpackage("multitask")
    config.add_subpackage("optimizers")
    config.add_subpackage("randombag")
    config.add_subpackage("ridge2")
    config.add_subpackage("rvfl")
    config.add_subpackage("simulation")
    config.add_subpackage("simulator")

    # submodules which have their own setup.py
    config.add_subpackage("optimizers")
    config.add_subpackage("sampling")

    # add the test directory
    config.add_subpackage("tests")

    # Skip cythonization as we do not want to include the generated
    # C/C++ files in the release tarballs as they are not necessarily
    # forward compatible with future versions of Python for instance.
    if "sdist" not in sys.argv:
        cythonize_extensions(top_path, config)

    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup

    setup(**configuration(top_path="").todict())
