import os


def configuration(parent_package="", top_path=None):
    import numpy
    from numpy.distutils.misc_util import Configuration

    config = Configuration("simulator", parent_package, top_path)

    libraries = []
    if os.name == "posix":
        libraries.append("m")

    # cpp_args = ['-stdlib=libc++', '-mmacosx-version-min=10.7']

    config.add_extension(
        "_simulatorc",
        sources=["_simulatorc.pyx", "simulator.cpp"],
        include_dirs=numpy.get_include(),
        libraries=libraries,
        language="c++",
        # extra_compile_args = cpp_args,
    )

    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup

    setup(**configuration(top_path="").todict())
