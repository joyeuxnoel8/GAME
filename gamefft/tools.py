"""
File name  : tools.py
Description:
Author     : Alioune Schurz
Lab        : Computational Molecular Design and Detection Lab (CMDD lab)
Institute  : National Taiwan University
"""

import time 
import sys,os

# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
# * * * * * * * * * * * * * *   Various   * * * * * * * * * * * * * * * * * * * * * *

def irange(a,b):
	return range(a,b+1)

def padding(x,size,elem):
	while(len(x)<size):
		x.append(elem)
	return x

def section(char='-',repeat=100):
    sec=""
    for i in range(0,repeat):
        sec+=char
    return sec
    
class Timer:
	def __init__(self):
		self.start_time=0
		self.elapsed=0
		self.stopped=True

	def __str__(self):
		return "Elapsed: "+str(self.elapsedTime())+" s"

	def start(self):
		self.start_time=time.time()
		self.stopped=False

	def stop(self):
		self.stopped=True

	def elapsedTime(self):
		if not self.stopped:
			self.elapsed=time.time()-self.start_time
		return self.elapsed

	class TimeoutException(Exception):
		def __init__(self, time):
			message="Timeout exception: "+str(time)
			Exception.__init__(self, message)
			self.time=time


# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
# * * * * * * * * * * * * * *   Parsers   * * * * * * * * * * * * * * * * * * * * * *


def weave_parse(filename):
	'''
	Parses a .weave.cc file with the following convention:
	//@headers
		... #include <xxxx>
	//@end
	//@support
		... declarations 
	//@end
	//@code var1,var2,var4, ... (variables shared with python script)
		... the main code 
	//@end
	Args:
		string filename  : the input file in .weave.cc format 
	Returns:
		headers,support_code,code,args (used for weave.inline function)
	'''
	args=[]
	headers=[]
	support_code=""
	code=""
	f=open(filename,'r')

	#checking that the file contains the right number 

	parsing_headers=False
	parsing_arg=False
	parsing_support_code=False
	parsing_code=False


	for line in f:
		line=line.strip()
		if line=="//@headers" and not parsing_headers:
				parsing_headers=True
				continue
		if parsing_headers and  "@end" in line :
				parsing_headers=False
				continue
		if line=="//@support" and not parsing_support_code:
				parsing_support_code=True
				continue
		if parsing_support_code and "@end" in line :
				parsing_support_code=False
				continue

		if "//@code" in line and not parsing_code:
				parsing_code=True
				line=line.replace("//@code","")
				line=line.strip()
				args=line.split(",")
				continue
		if parsing_code and  "@end" in line:
				parsing_code=False
				continue

		if parsing_headers and "#include" in line:
			line=line.replace("#include","")
			line=line.strip()
			headers.append(line)


		if parsing_code:
			if line:
				code+=line+"\n"

		if parsing_support_code:
			if line:
				support_code+=line+"\n"
	f.close() 
	return headers,list([sys.path[0]]),support_code,code,args

