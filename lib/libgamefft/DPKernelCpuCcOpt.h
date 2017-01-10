#ifndef DP_KERNEL_CPU_CC_OPT_H
#define DP_KERNEL_CPU_CC_OPT_H

#include <iostream>
#include <vector>
#include <string.h>
#include <algorithm>

typedef unsigned short uint16;
typedef unsigned int uint32;
typedef short int16;

#ifdef __cplusplus
extern "C" {
#endif
// code starts here

int DPKernelCpuCcOpt(	int,int,int,
						int,int*,int*,float*,
						uint16*,float*);

int DPKernelCpuCcOptBubble( 	int,int,int,
								int,int*,int*,float*,
								uint16*,float*);

int DPKernelCpuCcOptBubbleCompressed( 	int,int,int,
						int,int*,int*,float*,
						uint32*,float*,int*,int*);

// code ends here
#ifdef __cplusplus
}
#endif
#endif
