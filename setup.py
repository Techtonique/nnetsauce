import platform
from setuptools import setup, find_packages
from codecs import open
from os import path

__version__ = '0.26.0'

# get the dependencies and installs
here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')

install_all_requires = [
    x.strip() for x in all_reqs if "git+" not in x
]

dependency_links = [
    x.strip().replace("git+", "")
    for x in all_reqs
    if x.startswith("git+")
]

#if platform.system() in ('Linux', 'Darwin'):
#    install_jax_requires = ['jax', 'jaxlib']  
#else:
#    install_jax_requires = []

install_jax_requires = []

install_requires = [item for sublist in [install_all_requires, install_jax_requires] for item in sublist]

setup(
    name='nnetsauce',
    version=__version__,
    description='Quasi-randomized (neural) networks',
    long_description='Quasi-randomized (neural) networks for regression, classification and time series forecasting',
    url='https://techtonique.github.io/nnetsauce/',
    #alias='nnetsauce',
    download_url='https://github.com/Techtonique/nnetsauce',
    license='BSD Clause Clear',
    classifiers=[
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Developers',
      'Programming Language :: Python :: 3',
    ],
    keywords="",
    packages=find_packages(exclude=["docs", "tests*"]),
    include_package_data=True,
    author='T. Moudiki',
    install_requires=install_requires,
    dependency_links=dependency_links,
    author_email='thierry.moudiki@gmail.com'
)
