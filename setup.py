
from setuptools import setup, find_packages
from os import path


with open("README.md", "r") as fh:
    long_description = fh.read()
    
# get the dependencies and installs
here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')

install_requires = [x.strip() for x in all_reqs if 'git+' not in x]

setup(
    name="nnetsauce",
    version="0.5.0",
    url="https://github.com/thierrymoudiki/nnetsauce",
    packages=find_packages(),
    author="Thierry Moudiki",
    author_email="thierry.moudiki@gmail.com",
    description="Machine Learning using combinations of Neural Networks' layers",
    download_url="https://github.com/thierrymoudiki/nnetsauce",
    install_requires=["jax>=0.1.72", "jaxlib>=0.1.51", "joblib>=0.13.2", 
                      "numpy>=1.13.0", "rpy2>=3.2.2", "scipy>=0.19.0", 
                      "scikit-learn>=0.18.0", "six>=1.12.0", "tqdm>=4.28.1"].append(install_requires),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
)
