import os
from setuptools import setup

setup(
    name='histoptimizer',
    version='0.1',
    py_modules=['histoptimizer'],
    install_requires=[
        'Click',
    ],
    entry_points='''
        [console_scripts]
        histoptimizer=histoptimizer:histoptimizer_cli
    ''',
)


import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="histoptimizer",
    version="0.0.1",
    author="Kelly Joyner",
    author_email="de@lusion.org",
    description=("A library for creating even partitions of ordered items."),
    license="BSD",
    keywords = "histogram partition",
    url="https://github.com/delusionary/histoptimizer",
    packages=['histoptimizer', 'tests'],
    long_description=read('README'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
    install_requires=[
        'Click',
    ],
    entry_points='''
        [console_scripts]
        histoptimizer=histoptimizer:histoptimizer_cli
    ''',
)