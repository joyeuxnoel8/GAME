from gamefft.solvers.dp_solvers import AbstractDPKernelCompressed
from gamefft.solvers.dp_solvers import Encoder1
from gamefft.solvers.dp_solvers import GenericDPSolverOptCompressed
from jinja2 import Template
from numpy import *
from pycuda.compiler import SourceModule
import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import pycuda.tools
import os

# Kernels ========================================


class AbstractContextManager():
    
    class ScarceRessourceException(Exception):
        def __init__(self,res):
            msg ="\n===================================\n"
            msg+="Error: CUDA context error\n"
            for item in res:
                msg+=item+"\n"
            Exception.__init__(self,msg)
    
    def __init__(self):
        self.device=None
        self.shmemsize=0
        self.num_threads=0 
        self.block_dim=0 
        self.globalmemsize=0 
        self.constmemsize=0 
    
    def set_device(self,device):
        self.device=device
    
    def check(self):
        res=[]
        ok=False
        if self.device!=None:
            ok=True
            ok= self.shmemsize  <=self.device.MAX_SHARED_MEMORY_PER_BLOCK and ok
            ok= self.num_threads<=self.device.MAX_THREADS_PER_BLOCK and ok
            ok= self.block_dim <=self.device.MAX_GRID_DIM_X and ok
            ok= self.globalmemsize <= self.device.total_memory() and ok
            ok= self.constmemsize <= self.device.TOTAL_CONSTANT_MEMORY and ok
            res.append("shmemsize:  used %d bytes  [max: %d bytes]" % (self.shmemsize,self.device.MAX_SHARED_MEMORY_PER_BLOCK))
            res.append("numthreads: used %d  [max: %d]" % (self.num_threads,self.device.MAX_THREADS_PER_BLOCK))
            res.append("numblocks: used %d  [max: %d]" % (self.block_dim,self.device.MAX_GRID_DIM_X))
            res.append("globalmem: used %d MB  [max: %d MB]" % (self.globalmemsize //1024//1024,self.device.total_memory()//1024//1024))
            res.append("const mem: used %d bytes  [max: %d bytes]" % (self.constmemsize,self.device.TOTAL_CONSTANT_MEMORY))
        else:
            raise Exception("*ContextManager.check(): Warning: make sure to use set_device() method first !")
        return (ok,res)    
    
    def global_memory_usage(self):
        return float(self.globalmemsize)/float(self.device.total_memory())
    
class DefaultContextManager(AbstractContextManager):
    
    def __init__(self):
        AbstractContextManager.__init__(self)
        
    def calculate_ressources(self, R, wt_max, Ks, K_bytes, W_bytes, P_bytes, L_bytes, C_bytes, K_offsets_bytes, R_offsets_bytes):
        if self.device!=None:
            self.num_threads=None
            self.shmemsize=None
            t=int(log2(self.device.MAX_SHARED_MEMORY_PER_BLOCK/(4*3*Ks*R)))
            self.num_threads=int(2**t)
            self.shmemsize=int(4*3*Ks*R*self.num_threads)
            self.block_dim=int(sqrt(wt_max/self.num_threads)+1)
            self.constmemsize=0
            self.globalmemsize = 2*C_bytes+L_bytes+P_bytes+W_bytes+K_bytes+K_offsets_bytes+R_offsets_bytes
        else:
            raise Exception("*ContextManager.calculate_ressources(): Warning: make sure to use set_device() method first !")        
    
class DPKernelGpuCudaOptCompressed(AbstractDPKernelCompressed):
    
    def __init__(self,contextMngr=DefaultContextManager()):
        self.contextManager=contextMngr
    
    def run(self, n, R, wt_max, K, W, P, L, C, K_encoder, R_encoder, run_options):
      
        # generate kernel
        tpl = Template("""
        __device__ void fswap(float*,float*);
        __device__ void encode_at(const int*,unsigned int*,const unsigned int,const int);
        __device__ void iswap(unsigned int*,unsigned int*);
        
        
        extern __shared__  float  shmem32[];
        
        __global__ void DPKernelGpuCuda(float* C,float* Cprev,unsigned int* L,
                                        const float* P,const int* W,const int* K,
                                        const int* K_offsets,const int* R_offsets,
                                        int l,
                                        int s,int R,int wt_max)
        {
            const int w      = (gridDim.x*blockDim.x)*blockIdx.y + blockDim.x*blockIdx.x + threadIdx.x;
        
            if (w<=wt_max)
            {
        
                const int offset = 3*K[s]*R*threadIdx.x;
        
                unsigned int*   A_k = (unsigned int*)&shmem32[offset];
                unsigned int*   A_r = (unsigned int*)&shmem32[offset+K[s]*R];
                float*             A_p = &shmem32[offset+2*K[s]*R];
        
                int idx=0;
                for (int k = 0; k < K[s]; ++k)
                {
                    int sk=s*l+k;
        
                    if (w-W[sk]<0)
                        continue;
                    for (int r = 0; r < R; ++r) {
                        if(s>0)
                        {
                            A_k[idx]=k;A_r[idx]=r;
                            A_p[idx]=P[sk]*Cprev[(w-W[sk])*R+r];
                            idx++;
                        }else
                        {
                            if (r==1 && w-W[sk]==0)
                            {
                                A_k[idx]=k;A_r[idx]=r;
                                A_p[idx]=P[sk];
                                idx++;
                            }
                        }
                    }
                }
        
        
                //sort by probability (bubble sort)
                int n=idx;
                do {
                    int newn=0;
                    for (int i = 1; i < idx; ++i) {
                        if(A_p[i-1]<A_p[i])
                        {
                            iswap(&A_k[i],&A_k[i-1]);
                            iswap(&A_r[i],&A_r[i-1]);
                            fswap(&A_p[i],&A_p[i-1]);
                            newn=i;
                        }
                    }
                    n=newn;
                } while (n>0);
        
                //update C and L
                for (int r = 0; r < min(R,idx); ++r) {
                    int wr=w*R+r;
                    C[wr]=A_p[r];
                    wr=w*R*2+r*2;
                    encode_at(K_offsets,&L[wr+0],A_k[r],s);
                    encode_at(R_offsets,&L[wr+1],A_r[r],s);
                }
        
            }
        }
        
        __device__ void iswap(unsigned int* a,unsigned int* b)
        {
            int tmp=*b;
            *b=*a;
            *a=tmp;
        }
        
        
        __device__ void fswap(float* a,float* b)
        {
            float tmp=*b;
            *b=*a;
            *a=tmp;
        }
        
        __device__ void encode_at(const int* offsets, unsigned int* result, const unsigned int val, const int i)
        {
            *result = (*result) | val << offsets[i];
        }
        """)
        code=tpl.render()
        mod = SourceModule(code) 
        DPKernelGpuCuda = mod.get_function("DPKernelGpuCuda")
        argtypes=[  intp,intp,intp,
                  intp,intp,intp,
                  intp,intp,int32,
                  int32,int32,int32]
        #prepare cpu data =======================================      
        K_offsets = array(K_encoder.offsets, dtype=int32)
        R_offsets = array(R_encoder.offsets, dtype=int32)
        l = int(max([len(x) for x in W]))
        
        #check context ===========================================
        drv.init()
        dev = drv.Device(0)
        self.contextManager.set_device(dev)
        self.contextManager.calculate_ressources(R, wt_max, max(K),K.nbytes, W.nbytes, P.nbytes, L.nbytes, C.nbytes, K_offsets.nbytes, R_offsets.nbytes)
        ok,res=self.contextManager.check()    

        if not ok:
            raise AbstractContextManager.ScarceRessourceException(res)
        
        if run_options["verbose"]:
            print "========================================="
            print "GPU CONTEXT INFORMATION"
            for item in res:
                print item
            print "========================================="

        #Transfer data to Gpu
        C_gpu=gpuarray.zeros(C.shape,float32)
        Cprev_gpu=gpuarray.zeros(C.shape,float32)
        L_gpu=gpuarray.zeros(L.shape,uint32)
        P_gpu=gpuarray.to_gpu(P.astype(float32))
        W_gpu=gpuarray.to_gpu(W.astype(int32))
        K_gpu=gpuarray.to_gpu(K.astype(int32))
        K_offsets_gpu=gpuarray.to_gpu(K_offsets.astype(int32))
        R_offsets_gpu=gpuarray.to_gpu(R_offsets.astype(int32))

        #Run algorithm for s=0...n-1     
        Cswi=[C_gpu,Cprev_gpu]

        for s in range(0,n):
            #Run GPU kernel
            self.contextManager.calculate_ressources(R, wt_max,K[s],K.nbytes, W.nbytes, P.nbytes, L.nbytes, C.nbytes, K_offsets.nbytes, R_offsets.nbytes)
            DPKernelGpuCuda(  Cswi[s%2],Cswi[(s+1)%2],L_gpu,
                              P_gpu,W_gpu,K_gpu,
                              K_offsets_gpu,R_offsets_gpu,
                              int32(l),
                              int32(s),int32(R),int32(wt_max),
                              grid=(self.contextManager.block_dim,self.contextManager.block_dim,1),
                              block=(self.contextManager.num_threads,1,1),
                              shared=self.contextManager.shmemsize) 
   
        #copy back to CPU
        Cswi[(n-1)%2].get(C)
        L_gpu.get(L)   
        TIMEOUT = 0  
        SIGINT = 0
        return (SIGINT, TIMEOUT)   


# Specialized solvers ===============================

class DPSolverGpuCudaOptCompressed(GenericDPSolverOptCompressed):
    def __init__(self,contextMngr=DefaultContextManager()):
        GenericDPSolverOptCompressed.__init__(self, DPKernelGpuCudaOptCompressed(contextMngr))
