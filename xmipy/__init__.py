"""
Package for the eXtendend Model Interface.
It provides abstract classes, as well as the implementation `XmiWrapper`
"""

# exports
from bmipy.bmi import Bmi, BmiExtension

from xmipy.extensions import PointersExtension, XmiExtension
from xmipy.xmi import Xmi
from xmipy.xmiwrapper import XmiWrapper

__all__ = [
    "Bmi",
    "BmiExtension",
    "PointersExtension",
    "Xmi",
    "XmiExtension",
    "XmiWrapper",
]

__version__ = "2.0.0a1"
