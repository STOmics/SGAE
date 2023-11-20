from setuptools import Command, find_packages, setup

__lib_name__ = "SGAE"
__lib_version__ = "1.0.0"
__description__ = "Spatial domain deciphering with SGAE"
__url__ = "https://github.com/STOmics/SGAE.git"
__author__ = "Chao Yang"
__author_email__ = "yangchao4@genomics.cn"
__license__ = "MIT"
__keywords__ = ["Spatial transcriptomics", "Spatial clustering"]
__requires__ = ["requests", ]

setup(
    name=__lib_name__,
    version=__lib_version__,
    url=__url__,
    author=__author__,
    author_email=__author_email__,
    license=__license__,
    packages=["SGAE"],
    install_requires=__requires__,
    zip_safe=False,
    include_package_data=True,
    long_description="""SGAE is a graph neural network based method for spatial domain deciphering.""",
)
