from setuptools import setup, find_packages


try:
    with open("README.md", 'r') as f:
        long_description = f.read()

except:
    long_description = ""


setup(
    name='cidan',
    version='0.1.36',
    description='cidan-Calcium Imaging Data ANalysis',
    license="MIT",
    long_description=long_description,

    long_description_content_type="text/markdown",
    author='Sam Schickler',
    author_email='sschickl@ucsd.edu',
    # url="http://www.foopackage.com/",
    packages=find_packages(),
    install_requires=["numpy", "QtPy", "QDarkStyle", "pybind11", "pyqtgraph==0.11.0rc0",
                      "pyside2",
                      "dask[complete]", "matplotlib", "scipy", "tiffile",
                      "scikit-image", "pybind11", "hnswlib", "pillow",
                      'tifffile', "zarr", "neurofinder", "sklearn", 'pandas', 'future',
                      "peakutils", "requests"],
    include_package_data=True,

    entry_points={
        'console_scripts': [
            'cidan = cidan.__main__:main',
        ]
    },
)
