
import platform
from setuptools import setup, find_packages
from os import path


with open("README.md", "r") as fh:
    long_description = fh.read()
    
# get the dependencies and installs
here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')

install_requires = [x.strip() for x in all_reqs if 'git+' not in x]

install_jax_requires = ["jax>=0.1.72", "jaxlib>=0.1.51"] if platform.system() in ('Linux', 'Darwin') else []

setup(
    name="nnetsauce",
    version="0.5.2",
    url="https://github.com/thierrymoudiki/nnetsauce",
    packages=find_packages(),
    author="Thierry Moudiki",
    author_email="thierry.moudiki@gmail.com",
    description="Machine Learning using combinations of Neural Networks' layers",
    download_url="https://github.com/thierrymoudiki/nnetsauce",
    install_requires=install_jax_requires.append(install_requires),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
)
