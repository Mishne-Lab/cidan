from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
    name='LSSC-python',
    version='0.1',
    description='A module that creates ',
    # license="MIT", TODO add licence
    long_description=long_description,
    author='Sam Schickler',
    author_email='sschickl@ucsd.edu',
    # url="http://www.foopackage.com/",
    packages=['LSSC', "LSSC.functions"],  # same as name
    install_requires=[],  # TODO add external packages as dependencies
    scripts=[
        'LSSC/Stack.py',
        'LSSC/Parameters.py',
        'LSSC/Stack_Wrapper.py',
        'LSSC/functions/pickle_funcs.py',
        'LSSC/functions/data_manipulation.py',
        'LSSC/functions/embeddings.py',
        'LSSC/functions/roi_extraction.py',
    ]
)
