"""
PyQt Version Control
"""
from __future__ import absolute_import

__author__ = "Juhyeoung Kang"
__email__ = "jhkang@astro.snu.ac.kr"

from inspect import isclass
from PyQt5 import QtGui, QtPrintSupport, QtWidgets
from sys import modules, version_info

def class_name (module):
    """Extract class name of a given module."""
    dvalue = module.__dict__.values()
    a=[]
    for t in dvalue:
        try:
            if t and type(t) != int and type(t) != str and isclass(t) and t.__name__.startswith("Q"):
                a+=[t.__name__]
        except:
            pass
    return tuple(sorted(a))


VCQtWidgets=class_name(QtWidgets)
VCQtPrintSupport=class_name(QtPrintSupport)

def VCQtGui(qt):
    info={VCQtWidgets: qt.QtWidgets,
          VCQtPrintSupport: qt.QtPrintSupport,
          }
    
    if version_info[0] == 2:
        for cls_list, module in info.iteritems():
            for cls_name in cls_list:
                cls=getattr(module,cls_name)
                setattr(qt.QtGui,cls_name,cls)
                
    elif version_info[0] >= 3:
        for cls_list, module in info.items():
            for cls_name in cls_list:
                cls=getattr(module,cls_name)
                setattr(qt.QtGui,cls_name,cls)

VCQtGui(modules['PyQt5'])

