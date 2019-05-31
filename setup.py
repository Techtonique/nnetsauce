from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="nnetsauce",
    version="0.1.0",
    url="https://github.com/thierrymoudiki/nnetsauce",
    packages=find_packages(),
    author="Thierry Moudiki",
    author_email="thierry.moudiki@gmail.com",
    description="Machine Learning using combinations of Neural Networks' layers",
    download_url="https://github.com/thierrymoudiki/nnetsauce",
    install_requires=["numpy >= 1.13.0", "scipy >= 0.19.0", "scikit-learn >= 0.18.0"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
)
