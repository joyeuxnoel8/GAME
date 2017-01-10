from dp_solvers_cuda import DPSolverGpuCudaOptCompressed,DPKernelGpuCudaOptCompressed
from dp_solvers import  DPSolverCpuCcOpt 
import gamefft.npsdb.data_handler as dat
from gamefft.tools import Timer

run_options = {"verbose":False, "debug":False, "forcecompile":False, "maxruntime":float('Infinity'), "dec":3}

#create problem
scaffold=dat.Scaffold()
scaffold.load_data("s0000000001",run_options)
csccp=scaffold.generate_csccp(4,200, 5, 5) #414.304985

#solve problem
solverCUDA=DPSolverGpuCudaOptCompressed()
solverCPU=DPSolverCpuCcOpt()
timer=Timer()
for D in range(0,6):
    print "*****************"
    print "D=",D
    run_options["dec"]=D
    try:
        timer.start()
        solverCPU.run(csccp,run_options)
        print "solverCPU: ",timer.elapsedTime()," s"
        timer.start()
        solverCUDA.run(csccp,run_options)
        print "solverCUDA: ",timer.elapsedTime()," s"
        timer.stop()
    except Exception as e:
        print e

