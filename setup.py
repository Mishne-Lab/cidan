from setuptools import setup, find_packages


try:
    with open("README.md", 'r') as f:
        long_description = f.read()

except:
    long_description = ""


setup(
    name='cidan',
    version='0.1.27',
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
    scripts=[
        "cidan/LSSC/process_data.py",
        "cidan/TimeTrace/deltaFOverF.py",
        "cidan/TimeTrace/mean.py",
        "cidan/TimeTrace/waveletDenoise.py",
        "cidan/GUI/Tabs/AnalysisTab.py",
        "cidan/LSSC/SpatialBox.py",
        "cidan/TimeTrace/__init__.py",
        "cidan/LSSC/functions/roi_filter.py",
        'cidan/LSSC/functions/data_manipulation.py',
        "cidan/LSSC/functions/eigen.py",
        "cidan/LSSC/functions/embeddings.py",
        "cidan/LSSC/functions/pickle_funcs.py",
        "cidan/LSSC/functions/roi_extraction.py",
        "cidan/LSSC/functions/save_test_images.py",
        "cidan/LSSC/functions/temporal_correlation.py",
        "cidan/GUI/Console/__init__.py",
        "cidan/GUI/Console/ConsoleWidget.py",
        "cidan/GUI/Data_Interaction/__init__.py",
        "cidan/GUI/Data_Interaction/DataHandler.py",
        "cidan/GUI/Data_Interaction/PreprocessThread.py",
        "cidan/GUI/Data_Interaction/ROIExtractionThread.py",
        "cidan/GUI/Data_Interaction/Signals.py",
        "cidan/GUI/Data_Interaction/Thread.py",
        "cidan/GUI/Data_Interaction/loadDataset.py",
        "cidan/GUI/ImageView/__init__.py",
        "cidan/GUI/ImageView/ImageViewModule.py",
        "cidan/GUI/Inputs/__init__.py",
        "cidan/GUI/Inputs/Input.py",
        "cidan/GUI/Inputs/BoolInput.py",
        "cidan/GUI/Inputs/FileInput.py",
        "cidan/GUI/Inputs/FloatInput.py",
        "cidan/GUI/Inputs/Int3DInput.py",
        "cidan/GUI/Inputs/IntInput.py",
        "cidan/GUI/Inputs/OptionInput.py",
        "cidan/GUI/ListWidgets/__init__.py",
        "cidan/GUI/ListWidgets/ROIItemModule.py",
        "cidan/GUI/ListWidgets/ROIItemWidget.py",
        "cidan/GUI/ListWidgets/ROIListModule.py",
        "cidan/GUI/ListWidgets/TrialListWidget.py",
        "cidan/GUI/SettingWidget/__init__.py",
        "cidan/GUI/SettingWidget/SettingBlockModule.py",
        "cidan/GUI/SettingWidget/SettingsModule.py",
        'cidan/GUI/Tabs/__init__.py',
        "cidan/GUI/Tabs/FileOpenTab.py",
        "cidan/GUI/Tabs/PreprocessingTab.py",
        "cidan/GUI/Tabs/ROIExtractionTab.py",
        "cidan/GUI/Tabs/Tab.py",
        "cidan/__init__.py",
        "cidan/__main__.py"
    ],

    entry_points={
        'console_scripts': [
            'cidan = cidan.__main__:main',
        ]
    },
)
