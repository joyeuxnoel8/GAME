#!/usr/bin/env python
import sys

print "This program will check for required libraries..."
error=False

try:
	from numpy import *
except Exception as e:
	print e
try:
	from scipy import weave
except Exception as e:
	print e
try:
	import openbabel 
except Exception as e:
	print e
try:
	import pybel
except Exception as e:
	print e
try:
	import pycuda.driver as cuda
except Exception as e:
	print e
try:
	import pycuda.autoinit
except Exception as e:
	print e
try:
	from pycuda.compiler import SourceModule
except Exception as e:
	print e

