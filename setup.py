import os

from setuptools import setup


def read(fname):
    """
    Utility function to read the README file. Used for the long_description.
    It's nice, because now:
    1) we have a top level README file, and
    2) it's easier to type in the README file than to put a raw string in
    below.
    """
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="proteingraph",
    version="0.3.0",
    author="Eric J. Ma",
    author_email="ericmajinglong@gmail.com",
    description=(
        "A package for making graph representations of protein\
structures."
    ),
    license="MIT",
    keywords="protein, biochemistry, structural biology, graph theory",
    url="http://github.com/ericmjl/protein-interaction-network/",
    packages=["proteingraph"],
    long_description=read("README.md"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    include_package_data=True,
    long_description_content_type="text/markdown",
)
