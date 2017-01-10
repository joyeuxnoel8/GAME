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
		float* 			A_p = &shmem32[offset+2*K[s]*R];

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


