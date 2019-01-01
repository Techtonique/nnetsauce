from setuptools import setup, find_packages

setup(
    name='nnetsauce',
    version='0.1.0',
    url='https://github.com/thierrymoudiki/nnetsauce',
    packages=find_packages(),
    author='Thierry Moudiki',
    author_email="thierry.moudiki@gmail.com",
    description='Machine Learning using combinations of Single Layer Neural Networks',
#    download_url='https://github.com/thierrymoudiki/nnetsauce/tarball/0.6',
    install_requires=[
        "numpy >= 1.13.0",
        "scipy >= 0.19.0",
        "scikit-learn >= 0.18.0"]
)
