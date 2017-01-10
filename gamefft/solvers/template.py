from gamefft.npsdb.data_handler import CSCCP
from numpy import float64,float32,uint16,int16

class AbstractSolver:
	
	def __init__(self):
		self.float_t=float32
		self.int_t=uint16
		
	# @abstractmethod
	def run(self,csccp,options={}):
		return CSCCP.Result(None)
	

 