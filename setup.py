from setuptools import setup, find_packages

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
    name='CIDAN',
    version='0.1.2',
    description='CIDAN-Calcium Imaging Data ANalysis',
    license="MIT",
    long_description=long_description,
    author='Sam Schickler',
    author_email='sschickl@ucsd.edu',
    # url="http://www.foopackage.com/",
    packages=find_packages(),
    install_requires=["QtPy","QDarkStyle", "Pyside2","numpy", "dask[complete]", "matplotlib","scipy","tiffile","scikit-image","hnswlib", "pillow"],  # TODO add external packages as dependencies
    scripts=[
        "CIDAN/LSSC/process_data.py",
        "CIDAN/LSSC/SpatialBox.py",
        'CIDAN/LSSC/functions/data_manipulation.py',
        "CIDAN/LSSC/functions/eigen.py",
        "CIDAN/LSSC/functions/embeddings.py",
        "CIDAN/LSSC/functions/pickle_funcs.py",
        "CIDAN/LSSC/functions/roi_extraction.py",
        "CIDAN/LSSC/functions/save_test_images.py",
        "CIDAN/LSSC/functions/temporal_correlation.py",
        "CIDAN/ConsoleWidget.py",
        "CIDAN/main_stylesheet.css",
        "CIDAN/DataHandler.py",
        'CIDAN/DataHandlerWrapper.py',
        'CIDAN/fileHandling.py',
        'CIDAN/FileOpenTab.py',
        'CIDAN/ImageViewModule.py',
        'CIDAN/Input.py',
        'CIDAN/MainWindow.py',
        'CIDAN/PreprocessingTab.py',
        'CIDAN/ROIExtractionTab.py',
        'CIDAN/ROIItemModule.py',
        'CIDAN/ROIListModule.py',
        'CIDAN/roiTools.py',
        'CIDAN/SettingBlockModule.py',
        'CIDAN/SettingsModule.py',
        'CIDAN/Tab.py'
    ]
)
