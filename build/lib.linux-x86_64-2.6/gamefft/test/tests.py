from ctypes import *
from math import floor, log
import gamefft.npsdb.data_handler as dat
import gamefft.solvers.bf_solvers as bf_solvers
import gamefft.solvers.dp_solvers as dp_solvers
import gamefft.solvers.idp_solvers as idp_solvers
import gamefft.solvers.cuda.dp_solvers_cuda as cudp_solvers
import numpy as np
import os.path
import random
import sys, os
import unittest


"""
For remote testing, use command:
clear && xvfb-run python run-tests.py
"""

class ToleranceTestCase(unittest.TestCase):
    
    def assertScalarEqualDecimalTolerance(self, epsilon, first, second, msg):
        if abs((first - second) / first) > epsilon:
            raise AssertionError(msg)
        
        
    def assertScalarArrayEqualDecimalTolerance(self, epsilon, first, second, msg):
        for i, x in enumerate(first):
            self.assertScalarEqualDecimalTolerance(epsilon, x, second[i], msg)
    
class ResultValidationTest(ToleranceTestCase):
    
    def setUp(self):
        self.csccps = []
        try:
            scaffold = dat.Scaffold()
            path = os.path.abspath('')
            scaffold.load_data(path + "/_misc/data/small_set/s0000000001", {"verbose":True, "debug":False})
            self.csccps.append(scaffold.generate_csccp(4, 414.304985, 5, 5))
            
        except IOError as e:
            print "Error: The file %s was not found" % e.filename
            exit()  
      
    def compareSolvers(self, solver1, solver2, name1, name2, csccps, run_options):
        epsilon = 1e-3
        for csccp in csccps:
            tmp1 = solver1.run(csccp, run_options)
            tmp2 = solver2.run(csccp, run_options) 
            strcmp = name1 + ":\n" + str(tmp1) + "\n" + name2 + ":\n" + str(tmp2)
            
            if len(tmp1) != len(tmp2):
                print strcmp
            self.assertEqual(len(tmp1), len(tmp2), "%s and %s don't have same number of results")
            param = "%s -c %d -m %f --ppm %f" % (csccp.scaffold_name, csccp.configIdx, csccp.mass_peak, csccp.ppm)
            probs1, probs2 = [x[0] for x in tmp1], [x[0] for x in tmp2]
            weights1, weights2 = [x[1] for x in tmp1], [x[1] for x in tmp2]
            comps1, comps2 = [x[2] for x in tmp1], [x[2] for x in tmp2]
            
            
            for i, x in enumerate(tmp1):
                self.assertScalarArrayEqualDecimalTolerance(epsilon, probs1, probs2, "\nResults #%d of %s and %s on scaffold %s are different:\n%s\n%s" % (i, name1, name2, csccp.scaffold_name, param, strcmp))
                self.assertScalarArrayEqualDecimalTolerance(epsilon, weights1, weights2, "\nResults #%d of %s and %s on scaffold %s are different:\n%s\n%s" % (i, name1, name2, csccp.scaffold_name, param, strcmp))
                self.assertEqual(comps1, comps2, "\nResults #%d of %s and %s on scaffold %s are different:\n%s\n%s" % (i, name1, name2, csccp.scaffold_name, param, strcmp))

    def check_test_validity(self):
        run_options = {"verbose":False, "debug":False, "forcecompile":False, "maxruntime":float('Infinity'), "dec":0}
        self.compareSolvers(bf_solvers.BfSolverCpuPy(), bf_solvers.BfSolverCpuPy(), "BfSolverCpuPy", "BfSolverCpuPy", self.csccps, run_options)

    def Encoder1(self):
        """
        Encode and decode:
        K=[12,20,2,10]
        examples
        val=[11,19,1,9] -> 0b10011100111011
        val=[1,1,1,1] -> 0b11000010001
        val=[6,10,0,7] -> 0b1110010100110
        val=[5,18,1,2] -> 0b101100100101
        
        K=[512,11,11,512]
        examples   
        val=[5,11,1,2] -> 0b1000011011000000101
        val=[511,10,10,511] -> 0b11111111110101010111111111
        
        K=[128,128,128,512]
        examples
        val=[100,10,10,511] -> 0b111111111000101000010101100100
        """
        encoder = dp_solvers.Encoder1()
        for n in range(2, 10):
            random.seed()
            while True:
                K = [random.randrange(0, 100) for x in range(0, n)]
                try:
                    encoder.setup(K, 32)
                    break
                except Exception:
                    continue
            Km1 = [k - 1 for k in K]
            self.assertEquals(Km1, encoder.decode_all(encoder.encode_all(Km1)))
            for j in range(0, 100):
                val = [random.randrange(0, k) for k in K]
                self.assertEquals(val, encoder.decode_all(encoder.encode_all(val)))

    # ======================================================================================================================================================================
    def test_IDPSolverCpyPy_VS_BfSolverCpuPy(self):
        run_options = {"verbose":False, "debug":False, "forcecompile":False, "maxruntime":float('Infinity'), "dec":0}
        self.compareSolvers(bf_solvers.BfSolverCpuPy(), idp_solvers.IDPSolverCpuPy(), "BfSolverCpuPy", "IDPSolverCpuPy", self.csccps, run_options)
    
    def test_IDPSolverCpuPyOpt_VS_BfSolverCpuPy(self):
        run_options = {"verbose":False, "debug":False, "forcecompile":False, "maxruntime":float('Infinity'), "dec":0}
        self.compareSolvers(bf_solvers.BfSolverCpuPy(), idp_solvers.IDPSolverCpuPyOpt(), "BfSolverCpuPy", "IDPSolverCpuPyOpt", self.csccps, run_options)
        
    def test_IDPSolverCpuCcOpt_VS_BfSolverCpuPy(self):
        run_options = {"verbose":False, "debug":False, "forcecompile":False, "maxruntime":float('Infinity'), "dec":0}
        self.compareSolvers(bf_solvers.BfSolverCpuPy(), idp_solvers.IDPSolverCpuCcOpt(), "BfSolverCpuPy", "IDPSolverCpuCcOpt", self.csccps, run_options)  

    def test_IDPSolverCpuCcOptBubble_VS_BfSolverCpuPy(self):
        run_options = {"verbose":False, "debug":False, "forcecompile":False, "maxruntime":float('Infinity'), "dec":2}
        self.compareSolvers(bf_solvers.BfSolverCpuPy(), idp_solvers.IDPSolverCpuCcOptBubble(), "BfSolverCpuPy", "DPSolverCpuCcOptBubble", self.csccps, run_options)

    def test_IDPSolverCpuCcOptBubbleCompressed_VS_BfSolverCpuPy(self):
        run_options = {"verbose":False, "debug":False, "forcecompile":False, "maxruntime":float('Infinity'), "dec":2}
        self.compareSolvers(bf_solvers.BfSolverCpuPy(), idp_solvers.IDPSolverCpuCcOptBubbleCompressed(), "BfSolverCpuPy", "DPSolverCpuCcOptBubbleCompressed", self.csccps, run_options)
            
    def test_IDPSolverCpuPyOptCompressed_VS_BfSolverCpuPy(self):
        run_options = {"verbose":False, "debug":False, "forcecompile":False, "maxruntime":float('Infinity'), "dec":0}
        self.compareSolvers(bf_solvers.BfSolverCpuPy(), idp_solvers.IDPSolverCpuPyOptCompressed(), "BfSolverCpuPy", "IDPSolverCpuPyOptCompressed", self.csccps, run_options)
    
    def test_DPSolverGpuCudaOptCompressed_VS_DPSolverCpuCcOptBubbleCompressed(self):
        run_options = {"verbose":False, "debug":False, "forcecompile":False, "maxruntime":float('Infinity'), "dec":3}
        self.compareSolvers(dp_solvers.DPSolverCpuCcOptBubbleCompressed(), cudp_solvers.DPSolverGpuCudaOptCompressed(), "DPSolverCpuCcOptBubbleCompressed", "DPSolverGpuCudaOptCompressed", self.csccps, run_options)      
         
    def _test_pycuda(self):
        import pycuda.driver as drv
        import pycuda.tools
        import pycuda.autoinit
        import numpy
        import numpy.linalg as la
        from pycuda.compiler import SourceModule
        from jinja2 import Template
        import pycuda.gpuarray as gpuarray

        
        tpl2 = Template("""
        
        extern __shared__ int smem[];
        
        __global__ void square_them(int* a,int l)
        {
            
            const int i = blockDim.x*blockIdx.x +l*threadIdx.x;
            int* tmp=&smem[l*threadIdx.x];
            for(int j=0;j<l;++j)
                tmp[j]=a[i+j];
            __syncthreads();
            for(int j=0;j<l;++j)
                tmp[j]*=2;
            __syncthreads();
            for(int j=0;j<l;++j)
                a[i+j]=tmp[j];
            
        }
        """)
        
        drv.init()
        dev = drv.Device(0)
        

        t=9
        nthr=256
        size_arrays=1024*512
        num_blocks=64
        l=int(size_arrays/num_blocks/nthr)
        print l       
        a = range(0,l)
        a = a*(nthr*num_blocks)
        a = numpy.array(a, dtype=numpy.int32)    
        a_gpu=gpuarray.to_gpu(a.astype(numpy.int32))

        code = tpl2.render()
        #print code
        mod = SourceModule(code) 
        square_them = mod.get_function("square_them")
        shmem=4*l*nthr
        print "sh mem usage: ",shmem
        if shmem > dev.MAX_SHARED_MEMORY_PER_BLOCK:
            print "too much shared memory used"
            #return 
        print a[0:2*l]
        square_them(a_gpu,numpy.int32(l),grid=(num_blocks,1,1),block=(nthr, 1, 1),shared=shmem)
        print a[0:2*l]
    

    # ======================================================================================================================================================================

    def test_DPSolverCpuPy_VS_BfSolverCpuPy(self):
        run_options = {"verbose":False, "debug":False, "forcecompile":False, "maxruntime":float('Infinity'), "dec":2}
        self.compareSolvers(bf_solvers.BfSolverCpuPy(), dp_solvers.DPSolverCpuPy(), "BfSolverCpuPy", "DPSolverCpuPy", self.csccps, run_options)
    
    def test_DPSolverCpuPyOpt_VS_BfSolverCpuPy(self):
        run_options = {"verbose":False, "debug":False, "forcecompile":False, "maxruntime":float('Infinity'), "dec":2}
        self.compareSolvers(bf_solvers.BfSolverCpuPy(), dp_solvers.DPSolverCpuPyOpt(), "BfSolverCpuPy", "DPSolverCpuPyOpt", self.csccps, run_options)

    def test_DPSolverCpuCcOpt_VS_BfSolverCpuPy(self):
        run_options = {"verbose":False, "debug":False, "forcecompile":False, "maxruntime":float('Infinity'), "dec":2}
        self.compareSolvers(bf_solvers.BfSolverCpuPy(), dp_solvers.DPSolverCpuCcOpt(), "BfSolverCpuPy", "DPSolverCpuCcOpt", self.csccps, run_options)

    def test_DPSolverCpuPyOptCompressed_VS_BfSolverCpuPy(self):
        run_options = {"verbose":False, "debug":False, "forcecompile":False, "maxruntime":float('Infinity'), "dec":2}
        self.compareSolvers(bf_solvers.BfSolverCpuPy(), dp_solvers.DPSolverCpuPyOptCompressed(), "BfSolverCpuPy", "DPSolverCpuPyOptCompressed", self.csccps, run_options)
        
       

