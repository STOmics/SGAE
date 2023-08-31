#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 8/31/23 4:11 PM
# @Author  : yangchao
# @File    : setup.py.py
# @Email   : yangchao4@genomics.cn
import setuptools
from wheel.bdist_wheel import bdist_wheel

__version__ = "1.0.0"


class BDistWheel(bdist_wheel):
    def get_tag(self):
        return (self.python_tag, "none", "any")


cmdclass = {
    "bdist_wheel": BDistWheel,
}

requirements = open("requirements.txt").readline()

setuptools.setup(
    name="SGAE",
    version=__version__,
    author="yangchao",
    author_email="yangchao4@genomics.cn",
    url="https://github.com/STOmics/SGAE.git",
    description="SGAE",
    python_requires=">=3.8",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    cmdclass=cmdclass,
    package_data={'': ["*.html"]},
)