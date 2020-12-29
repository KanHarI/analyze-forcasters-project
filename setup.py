# Tested with python 3.8

from setuptools import setup, find_packages

__version__ = "0.1.0"

setup(
    name="AnalyzeForcastersProject",
    version=__version__,
    description="Analysis of the forecasters project",
    install_requires=['pandas', 'numpy', 'scipy', 'matplotlib'],
    entry_points={},
    author="Itay Knaan Harpaz",
    author_email="knaan.harpaz@gmail.com",
    url="https://github.com/KanHarI/analyze-forecasters-project",
)
