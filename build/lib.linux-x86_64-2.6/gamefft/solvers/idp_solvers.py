from numpy import *
from scipy import weave
import gamefft.npsdb.data_handler as dat
import gamefft.solvers.dp_solvers as dp_solvers 


# Generic solver ====================================
class GenericIDPSolver(dp_solvers.AbstractDPSolver):
    def __init__(self, kernel):
        dp_solvers.AbstractDPSolver.__init__(self)
        self.kernel = kernel
        
    def run(self, csccp, run_options):

        if csccp.trivial():
            raise dat.TrivialCSCCP("This CSCCP problem is trivial")
        if run_options['verbose']:
            print "Starting Iterative Dynamic Programming"
    
        # 1- Variable init * * * * * * * * * * * * * * * * * * * * * * * * * * * 
        l = max([len(x) for x in csccp.W])
        n = csccp.n
        multiplier = 10
        R = csccp.R
        RR = multiplier * R
        wt_max = int(csccp.wmax + 0.5 * n)
        wt_min = int(csccp.wmin - 0.5 * n)
        K = array(csccp.K)
        W = array([([int(w) for w in wside] + [0] * l)[:l] for wside in csccp.W])
        P = array([(pside + [0] * l)[:l] for pside in csccp.P])   
             
        try:
            # 2- Core computations loop * * * * * * * * * * * * * * * * * * * * *  
            count = 0
            reason = "none"
            valid_candidates = []
            
            while True:
                count += 1
                # 2.1- Variable init for the loop
                C = zeros((n, wt_max + 1, RR), self.float_t)  # [0 .. n-1] x [0 .. wt_max] x [0 .. R-1] 
                L = zeros((n, wt_max + 1, RR, 2), self.int_t)
                
                # 2.2- Core computations     
                SIGINT, TIMEOUT = self.kernel.run(n, RR, wt_max, K, W, P, L, C, run_options)
                if SIGINT:    
                    raise KeyboardInterrupt
                if TIMEOUT > 0:
                    raise tools.Timer.TimeoutException(TIMEOUT)
    
                # 2.3- Result filtering
                len_filtered_results = 0
                for w in range(wt_min, wt_max + 1):
                    for r in range(0, RR):
                        probability = C[csccp.n - 1][w][r]
                        if probability:
                            len_filtered_results += 1
                            real_weight, comp = self.compute_real_weight(csccp, w, r, W, L)
                            if real_weight >= csccp.wmin and real_weight <= csccp.wmax:
                                valid_candidates.append((probability * csccp.scaffold_probability_rel2config, real_weight + csccp.scaffold_weight_rel2config, comp))
    
                # 2.4- Taking decision to iterate or not
                if len_filtered_results and len_filtered_results < RR:
                    reason = "len(filtered_results)=%d < RR" % (len_filtered_results)
                    break
                if len(valid_candidates) >= R:
                    reason = "found R valid candidates"
                    break    
                if RR >= csccp.number_of_compounds:    
                    break
                RR = min(RR * multiplier, csccp.number_of_compounds)
            # -- END WHILE
            
            if run_options["verbose"]:
                print "Finished with %d iterations: RR=%d<=%d. Reason: %s" % (count, RR, csccp.number_of_compounds, reason)
            valid_candidates = sorted(valid_candidates, key=lambda x:x[0], reverse=True)
            valid_candidates = valid_candidates[0:R]
            return dat.CSCCP.Result(array(valid_candidates, dtype=dtype([('p', float), ('w', float), ('comp', list)])))
    
        except KeyboardInterrupt:
            print "*  WARNING: the program was interrupted by user (CTRL+C). The program will now terminate *"
            exit()     
            
class GenericIDPSolverOpt(dp_solvers.AbstractDPSolver):
    
    def __init__(self, kernelOpt):
        dp_solvers.AbstractDPSolver.__init__(self)
        self.kernelOpt  = kernelOpt
    
    def run(self, csccp, run_options):

        if csccp.trivial():
            raise dat.TrivialCSCCP("This CSCCP problem is trivial")
        if run_options['verbose']:
            print "Starting Iterative Dynamic Programming"
    
        # 1- Variable init * * * * * * * * * * * * * * * * * * * * * * * * * * * 
        l = max([len(x) for x in csccp.W])
        n = csccp.n
        multiplier = 10
        R = csccp.R
        RR = multiplier * R
        wt_max = int(csccp.wmax + 0.5 * n)
        wt_min = int(csccp.wmin - 0.5 * n)
        K = array(csccp.K,dtype=int32)
        W = array([([int(w) for w in wside] + [0] * l)[:l] for wside in csccp.W],dtype=int32)
        P = array([(pside + [0] * l)[:l] for pside in csccp.P],dtype=float32)   
             
        try:
            # 2- Core computations loop * * * * * * * * * * * * * * * * * * * * *  
            count = 0
            reason = "none"
            valid_candidates = []
            
            while True:
                count += 1
                # 2.1- Variable init for the loop
                C = zeros((wt_max + 1, RR), self.float_t)  # [0 .. n-1] x [0 .. wt_max] x [0 .. R-1] 
                L = zeros((n,wt_max + 1, RR,2), self.int_t)
                
                # 2.2- Core computations 
                SIGINT, TIMEOUT = self.kernelOpt.run(n, RR, wt_max, K, W, P, L, C,run_options)
                if SIGINT:    
                    raise KeyboardInterrupt
                if TIMEOUT > 0:
                    raise tools.Timer.TimeoutException(TIMEOUT)
    
                # 2.3- Result filtering
                len_filtered_results = 0
                for w in range(wt_min, wt_max + 1):
                    for r in range(0, RR):
                        probability = C[w][r]
                        if probability>0:
                            len_filtered_results += 1
                            real_weight, comp = self.compute_real_weight(csccp, w, r, W, L)
                            if real_weight >= csccp.wmin and real_weight <= csccp.wmax:
                                valid_candidates.append((probability * csccp.scaffold_probability_rel2config, real_weight + csccp.scaffold_weight_rel2config, comp))
    
                # 2.4- Taking decision to iterate or not
                if len_filtered_results and len_filtered_results < RR:
                    reason = "len(filtered_results)=%d < RR" % (len_filtered_results)
                    break
                if len(valid_candidates) >= R:
                    reason = "found R valid candidates"
                    break    
                if RR >= csccp.number_of_compounds:    
                    break
                RR = min(RR * multiplier, csccp.number_of_compounds)
            # -- END WHILE
            
            if run_options["verbose"]:
                print "Finished with %d iterations: RR=%d<=%d. Reason: %s" % (count, RR, csccp.number_of_compounds, reason)
            valid_candidates = sorted(valid_candidates, key=lambda x:x[0], reverse=True)
            valid_candidates = valid_candidates[0:R]
            return dat.CSCCP.Result(array(valid_candidates, dtype=dtype([('p', float), ('w', float), ('comp', list)])))
    
        except KeyboardInterrupt:
            print "*  WARNING: the program was interrupted by user (CTRL+C). The program will now terminate *"
            exit()

class GenericIDPSolverOptCompressed(dp_solvers.AbstractDPSolverCompressed):
    
    def __init__(self, kernelOptCompressed):
        dp_solvers.AbstractDPSolverCompressed.__init__(self)
        self.kernelOptCompressed  = kernelOptCompressed
        self.int_t=uint32
    
    def run(self, csccp, run_options):

        if csccp.trivial():
            raise dat.TrivialCSCCP("This CSCCP problem is trivial")
        if run_options['verbose']:
            print "Starting Iterative Dynamic Programming C-optimized L-compressed"
    
        # 1- Variable init * * * * * * * * * * * * * * * * * * * * * * * * * * * 
        l = max([len(x) for x in csccp.W])
        n = csccp.n
        multiplier = 10
        R = csccp.R
        RR = multiplier * R
        wt_max = int(csccp.wmax + 0.5 * n)
        wt_min = int(csccp.wmin - 0.5 * n)
        K = array(csccp.K,dtype=int32)
        W = array([([int(w) for w in wside] + [0] * l)[:l] for wside in csccp.W],dtype=int32)
        P = array([(pside + [0] * l)[:l] for pside in csccp.P],dtype=float32)   
             
        try:
            # 2- Core computations loop * * * * * * * * * * * * * * * * * * * * *  
            count = 0
            reason = "none"
            valid_candidates = []
            
            K_encoder=dp_solvers.Encoder1()
            R_encoder=dp_solvers.Encoder1()
            K_encoder.setup(K,32)
            
            while True:
                count += 1
                # 2.1- Variable init for the loop
                C = zeros((wt_max + 1, RR), float32)  # [0 .. n-1] x [0 .. wt_max] x [0 .. R-1] 
                L = zeros((wt_max + 1, RR,2), uint32)
                R_encoder.setup([RR-1 for x in range(0,n)],32)
                
                # 2.2- Core computations     
                SIGINT, TIMEOUT = self.kernelOptCompressed.run(n, RR, wt_max, K, W, P, L, C, K_encoder, R_encoder, run_options)
                if SIGINT:    
                    raise KeyboardInterrupt
                if TIMEOUT > 0:
                    raise tools.Timer.TimeoutException(TIMEOUT)
                # 2.3- Result filtering
                len_filtered_results = 0
                for w in range(wt_min, wt_max + 1):
                    for r in range(0, RR):
                        probability = C[w][r]
                        if probability:
                            len_filtered_results += 1
                            real_weight, comp = self.compute_real_weight(csccp, w, r, W, L, K_encoder, R_encoder)
                            if real_weight >= csccp.wmin and real_weight <= csccp.wmax:
                                valid_candidates.append((probability * csccp.scaffold_probability_rel2config, real_weight + csccp.scaffold_weight_rel2config, comp))
    
                # 2.4- Taking decision to iterate or not
                if len_filtered_results and len_filtered_results < RR:
                    reason = "len(filtered_results)=%d < RR" % (len_filtered_results)
                    break
                if len(valid_candidates) >= R:
                    reason = "found R valid candidates"
                    break    
                if RR >= csccp.number_of_compounds:    
                    break
                RR = min(RR * multiplier, csccp.number_of_compounds)
            # -- END WHILE
            
            if run_options["verbose"]:
                print "Finished with %d iterations: RR=%d<=%d. Reason: %s" % (count, RR, csccp.number_of_compounds, reason)
            valid_candidates = sorted(valid_candidates, key=lambda x:x[0], reverse=True)
            valid_candidates = valid_candidates[0:R]
            return dat.CSCCP.Result(array(valid_candidates, dtype=dtype([('p', float), ('w', float), ('comp', list)])))
    
        except KeyboardInterrupt:
            print "*  WARNING: the program was interrupted by user (CTRL+C). The program will now terminate *"
            exit() 
                   
# Specialized solvers ===============================
class IDPSolverCpuPy(GenericIDPSolver):
    def __init__(self):
        GenericIDPSolver.__init__(self, dp_solvers.DPKernelCpuPy())

class IDPSolverCpuPyOpt(GenericIDPSolverOpt):
    def __init__(self):
        GenericIDPSolverOpt.__init__(self, dp_solvers.DPKernelCpuPyOpt())

class IDPSolverCpuCcOpt(GenericIDPSolverOpt):
    def __init__(self):
        GenericIDPSolverOpt.__init__(self, dp_solvers.DPKernelCpuCcOpt())
        
class IDPSolverCpuPyOptCompressed(GenericIDPSolverOptCompressed):
    def __init__(self):
        GenericIDPSolverOptCompressed.__init__(self, dp_solvers.DPKernelCpuPyOptCompressed())
        
class IDPSolverCpuCcOptCompressed(GenericIDPSolverOptCompressed):
    def __init__(self):
        GenericIDPSolverOptCompressed.__init__(self, dp_solvers.DPKernelCpuCcOptCompressed())
    
class IDPSolverCpuCcOptBubble(GenericIDPSolverOpt):
    def __init__(self):
        GenericIDPSolverOpt.__init__(self, dp_solvers.DPKernelCpuCcOptBubble())

class IDPSolverCpuCcOptBubbleCompressed(GenericIDPSolverOptCompressed):
    def __init__(self):
        GenericIDPSolverOptCompressed.__init__(self, dp_solvers.DPKernelCpuCcOptBubbleCompressed())