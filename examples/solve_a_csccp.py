#import module to handle scaffold files and CSCCPs
import gamefft.npsdb.data_handler as dat
#import different CSCCP solvers
import gamefft.solvers.bf_solvers as bf_solvers
import gamefft.solvers.dp_solvers as dp_solvers
import gamefft.solvers.idp_solvers as idp_solvers
import gamefft.solvers.cuda.dp_solvers_cuda as cudp_solvers
import string

SOLVER_TYPE=idp_solvers.IDPSolverCpuPyOpt
SCAFFOLD_FILE="example-data/s0000000001"

def main():
	run_options={"verbose":False, "debug":False, "forcecompile":False, "maxruntime":float('Infinity'), "dec":0}
	#load a scaffold file
	scaffold = dat.Scaffold()
	scaffold.load_data(SCAFFOLD_FILE,run_options)

	#create a csccp problem 
	R=4
	mass_peak=414.304985
	ppm=5
	configIdx=5
	csccp=scaffold.generate_csccp(R,mass_peak,ppm,configIdx)

	#instanciate the chosen solver
	solver=SOLVER_TYPE()
	tmp=solver.run(csccp, run_options)

	#print result before generating final molecules
	print "========================================"
	print "Result before generating final molecules\n",tmp

	#generate final result
	result=csccp.generate_output_molecules(tmp,run_options) 
	print "========================================"
	print "Result after generating final molecules\n",result

	#show ranking and SMILE's
	print "========================================"
	print "Final Ranking"
	for i,res in enumerate(result):
		SMILE=str(res['mol']).replace("_",'').replace("\n",'')
		print i+1,".\t",res['weight'],"\t",res['probability'],"\t",SMILE


if __name__=="__main__":
	main()


