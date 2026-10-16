"""
Package for the eXtendend Model Interface.
It provides abstract classes, as well as the implementation `XmiWrapper`
"""

# exports
from bmipy.bmi import Bmi

from xmipy.xmi import Xmi
from xmipy.xmiwrapper import XmiWrapper

__all__ = ["Bmi", "Xmi", "XmiWrapper"]

__version__ = "2.0.0a1"
