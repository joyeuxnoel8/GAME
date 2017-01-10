#include "DPKernelCpuCcOpt.h"

using namespace std;

#define min(a,b) ({ __typeof__ (a) _a = (a); __typeof__ (b) _b = (b); _a < _b ? _a : _b; })

extern "C" struct tuple
{
  		uint16 k;
  		uint16 r;
  		float p;

		tuple(uint k,uint r,float p) : k(k),r(r),p(p) {}

		inline bool operator < (const tuple& t) const
		{
			//the comparison operator is voluntarily reversed so that the array will be reverse sorted
        	return (t.p < p);
		}
};


inline void swap(uint16* A_k,uint16* A_r,float* A_p,uint16 i,uint16 j)
{
	uint16 tmp_k,tmp_r;
	float tmp_p;
	tmp_k=A_k[j];
	tmp_r=A_r[j];
	tmp_p=A_p[j];

	A_k[j]=A_k[i];
	A_r[j]=A_r[i];
	A_p[j]=A_p[i];

	A_k[i]=tmp_k;
	A_r[i]=tmp_r;
	A_p[i]=tmp_p;
};

inline uint32 encode_at(int* offsets, uint32 a, uint32 b, int i)
{
	return a | b << offsets[i];
}

int DPKernelCpuCcOpt( 	int n, int R, int wt_max,
						int l, int* K, int* W, float* P,
						uint16* L,float* C)
{//verified

	int C_len=(wt_max+1)*R;
	float* Cnew=new float[C_len];

	float* ptr_tmp;
	float* ptr_C=C;
	float* ptr_Cnew=Cnew;
	for (int s = 0; s < n; ++s) {
		//switch current C and Cnew
		ptr_tmp=ptr_C;
		ptr_C=ptr_Cnew;
		ptr_Cnew=ptr_tmp;
		for (int w = 1; w < wt_max+1; ++w) {
			vector<tuple> A;
			for (int k = 0; k < K[s]; ++k) {
				int sk=s*l+k;
				if (w-W[sk]<0)
					continue;
				for (int r = 0; r < R; ++r) {
					if(s>0){
						int idx=(w-W[sk])*R+r;
						float p=P[sk]*ptr_C[idx];
						A.push_back(tuple(k,r,p));
					}else if(r==1 && w-W[sk]==0){
						A.push_back(tuple(k,r,P[sk]));
					}
				}
			}
			//sort by probability
			std::sort(A.begin(),A.end());
			//update C and L
			for (int r = 0; r < min(R,A.size()); ++r) {
				int wr=w*R+r;
				ptr_Cnew[wr]=A[r].p;
				int swr=s*(wt_max+1)*R*2+w*R*2+r*2;
				L[swr+0]=A[r].k;
				L[swr+1]=A[r].r;
			}
		}
	}
	if (ptr_Cnew!=C)
		memcpy(C,ptr_Cnew,C_len*sizeof(float));
	delete Cnew;
	return 0;
}

int DPKernelCpuCcOptBubble( 	int n, int R, int wt_max,
						int l, int* K, int* W, float* P,
						uint16* L,float* C)
{

	int C_len=(wt_max+1)*R;
	float* Cnew=new float[C_len];

	float* ptr_tmp;
	float* ptr_C=C;
	float* ptr_Cnew=Cnew;

	for (int s = 0; s < n; ++s) {
		//switch current C and Cnew
		ptr_tmp=ptr_C;
		ptr_C=ptr_Cnew;
		ptr_Cnew=ptr_tmp;

		for (int w = 1; w < wt_max+1; ++w) {

			uint16 ptr_A=0;
			uint16 A_size=K[s]*R;
			uint16* A_k=new uint16[A_size];
			uint16* A_r=new uint16[A_size];
			float*  A_p=new float[A_size];

			for (int k = 0; k < K[s]; ++k) {
				int sk=s*l+k;
				if (w-W[sk]<0)
					continue;
				for (int r = 0; r < R; ++r) {
					if(s>0){
						int idx=(w-W[sk])*R+r;
						float p=P[sk]*ptr_C[idx];
						A_k[ptr_A]=k;A_r[ptr_A]=r;A_p[ptr_A]=p;
						ptr_A++;
					}else if(r==1 && w-W[sk]==0){
						A_k[ptr_A]=k;A_r[ptr_A]=r;A_p[ptr_A]=P[sk];
						ptr_A++;
					}
				}
			}
			//sort by probability (bubble sort)
			int n=ptr_A;
			do {
				int newn=0;
				for (int i = 1; i < ptr_A; ++i) {
					if(A_p[i-1]<A_p[i])
					{
						swap(A_k,A_r,A_p,i,i-1);
						newn=i;
					}
				}
				n=newn;
			} while (n>0);

			//update C and L
			for (int r = 0; r < min(R,ptr_A); ++r) {
				int wr=w*R+r;
				ptr_Cnew[wr]=A_p[r];
				int swr=s*(wt_max+1)*R*2+w*R*2+r*2;
				L[swr+0]=A_k[r];
				L[swr+1]=A_r[r];
			}
			delete A_k;
			delete A_r;
			delete A_p;
		}


	}
	if (ptr_Cnew!=C)
		memcpy(C,ptr_Cnew,C_len*sizeof(float));
	delete Cnew;
	return 0;
}

int DPKernelCpuCcOptBubbleCompressed( 	int n, int R, int wt_max,
						int l, int* K, int* W, float* P,
						uint32* L,float* C,int* K_offsets,int* R_offsets)
{
	int C_len=(wt_max+1)*R;
	float* Cnew=new float[C_len];

	float* ptr_tmp;
	float* ptr_C=C;
	float* ptr_Cnew=Cnew;

	for (int s = 0; s < n; ++s) {
		//switch current C and Cnew
		ptr_tmp=ptr_C;
		ptr_C=ptr_Cnew;
		ptr_Cnew=ptr_tmp;

		for (int w = 1; w < wt_max+1; ++w) {

			uint16 ptr_A=0;
			uint16 A_size=K[s]*R;
			uint16* A_k=new uint16[A_size];
			uint16* A_r=new uint16[A_size];
			float*  A_p=new float[A_size];

			for (int k = 0; k < K[s]; ++k) {
				int sk=s*l+k;
				if (w-W[sk]<0)
					continue;
				for (int r = 0; r < R; ++r) {
					if(s>0){
						int idx=(w-W[sk])*R+r;
						float p=P[sk]*ptr_C[idx];
						A_k[ptr_A]=k;A_r[ptr_A]=r;A_p[ptr_A]=p;
						ptr_A++;
					}else if(r==1 && w-W[sk]==0){
						A_k[ptr_A]=k;A_r[ptr_A]=r;A_p[ptr_A]=P[sk];
						ptr_A++;
					}
				}
			}
			//sort by probability (bubble sort)
			int n=ptr_A;
			do {
				int newn=0;
				for (int i = 1; i < ptr_A; ++i) {
					if(A_p[i-1]<A_p[i])
					{
						swap(A_k,A_r,A_p,i,i-1);
						newn=i;
					}
				}
				n=newn;
			} while (n>0);

			//update C and L
			for (int r = 0; r < min(R,ptr_A); ++r) {
				int wr=w*R+r;
				ptr_Cnew[wr]=A_p[r];
				wr=w*R*2+r*2;
				L[wr+0]=encode_at(K_offsets,L[wr+0],A_k[r],s);
				L[wr+1]=encode_at(R_offsets,L[wr+1],A_r[r],s);
			}
			delete A_k;
			delete A_r;
			delete A_p;
		}


	}
	if (ptr_Cnew!=C)
		memcpy(C,ptr_Cnew,C_len*sizeof(float));
	delete Cnew;
	return 0;
}
