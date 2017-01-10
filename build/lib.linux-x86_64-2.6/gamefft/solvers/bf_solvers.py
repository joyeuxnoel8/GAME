from gamefft.tools import *
import gamefft.solvers.template as template
import gamefft.npsdb.data_handler as dat
from numpy import *
from operator import mul

class BfSolverCpuPy(template.AbstractSolver):
		
	def run(self,csccp,options={}):
		'''
		Args:
			CSCCP csccp    : the input problem
			      options  : {verbose,debug,maxruntime} 
		Returns:
			CSCCP.Result: the results
		Raises:
			TimeoutException
		'''

		if csccp.trivial():
			raise dat.TrivialCSCCP("This CSCCP problem is trivial")

		if options["verbose"]:
			print "Starting python CPU Brute force solver..."
			
		res=[]
		minimum=1
		minIdx=0
		count=0
		timer=Timer()
		timer.start()
		
		# 2- Solve CSCCP and handle interruption and timeout exceptions
		try:
			for comp in csccp:
				#stop the program if it takes too long to run
				if timer.elapsedTime()>options["maxruntime"]:
					raise Timer.TimeoutException(timer.elapsedTime())

				weights =[0]+[csccp.W[s][comp[s]] for s in range(0,csccp.n)]
				weight  =sum(weights)
				if options["debug"]:
					print "Debug:",weight+csccp.scaffold_weight_rel2config,"(weight computed)"

				# we only keep coumpounds that have the right weight
				if weight>=csccp.wmin and weight<=csccp.wmax:
					# then we compute the probability
					probabilities=[1]+[csccp.P[s][comp[s]] for s in range(0,csccp.n)]
					probabilities=array(probabilities,dtype=self.float_t)
					probability=multiply.reduce(probabilities)
					#probability=reduce(mul,probabilities)

					if not len(res):
						minimum=probability
						minIdx=0
						res.append([probability,weight,comp])
					
					elif len(res)<csccp.R:
						if probability<minimum:
							minIdx=len(res)
							minimum=probability
						#if probability>=minimum: we just add it to the list without changing 
						res.append([probability,weight,comp])
					else:
						#if probability<minimum: we do nothing
						if probability>=minimum:
							res[minIdx]=[probability,weight,comp]
							minIdx=argmin([x[0] for x in res])
							minimum=res[minIdx][0]	
		except KeyboardInterrupt:
			print "* WARNING: the program was interrupted by user (CTRL+C). Results might be inaccurate! *"
		
		# 3- Processing results
		res=sorted(res,key=lambda x:x[0],reverse=True)
		res=array([(p*csccp.scaffold_probability_rel2config,w+csccp.scaffold_weight_rel2config,comp) for p,w,comp in res],dtype=dtype([('p',float),('w',float),('comp',list)]))
		if options["verbose"]:
			print "Done."
			
		return dat.CSCCP.Result(res)

"""
class BfSolverCpuCc(template.AbstractSolver):
	def run(self,csccp,options={}):
		'''
		Args:
			CSCCP csccp    : the input problem
			      options  : {verbose,debug,forcecompile,maxruntime} 
		Returns:
			CSCCP.Result: the results
		Raises:
			TimeoutException
		Bugs:
			Validity has to be checked !!! (SEG fault!! problems with iterators)
		'''
		if csccp.trivial():
			raise dat.TrivialCSCCP("This CSCCP problem is trivial")
		
		if options["verbose"]:
			print "Starting python C++ CPU Brute force solver..."
			
		# 1- Initialize variable shared with C code
		n=csccp.n
		R=csccp.R
		wmin=csccp.wmin
		wmax=csccp.wmax
		W =csccp.prepare_W()
		P=csccp.prepare_P()
		K=csccp.prepare_K()
		res=zeros((R,2+len(K)))
		MAX_RUNTIME=options["maxruntime"]

		# 2- Solve CSCCP with C code and handle interruption and timeout exceptions
		try:
			headers,include_dirs,support_code,code,arg_names=weave_parse("gamefft/solvers/bf_cpu.weave.cc")
			SIGINT,TIMEOUT=weave.inline(code,support_code=support_code,headers = headers,include_dirs=include_dirs,arg_names=arg_names,type_converters=weave.converters.blitz,force=options["forcecompile"],verbose=int(options["verbose"]))
			if SIGINT==1:
				raise KeyboardInterrupt
			if TIMEOUT>0:
				raise Timer.TimeoutException(TIMEOUT)

		except KeyboardInterrupt:
			print "* WARNING: the program was interrupted by user (CTRL+C). Results might be inaccurate! *"
		except IOError as e:
			print "Error: Make sure the c++ source file \"%s\" in the same directory as \"dp.py\"." % e.filename 
			exit()

		# 3- Processing results
		res=[tuple([x[0],x[1],list([int(i) for i in x[2:]])]) for x in res if x[0]>0]
		res=sorted(res,key=lambda x:x[0],reverse=True)
		res=array([(p*csccp.scaffold_probability_rel2config,w+csccp.scaffold_weight_rel2config,comp) for p,w,comp in res],dtype=dtype([('p',float),('w',float),('comp',list)]))
		
		if options["verbose"]:
			print "Done."
			
		return dat.CSCCP.Result(res)
"""

