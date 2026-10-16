"""
Package for the eXtendend Model Interface.
It provides abstract classes, as well as the implementation `XmiWrapper`
"""

# exports
from bmipy.bmi import Bmi, BmiExtension

from xmipy.extensions import XmiExtension
from xmipy.xmi import Xmi
from xmipy.xmiwrapper import XmiWrapper

__all__ = ["Bmi", "BmiExtension", "Xmi", "XmiExtension", "XmiWrapper"]

__version__ = "2.0.0a1"
