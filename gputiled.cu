#include "cx.h"
#include "cxtimers.h"
#include <random>
#include <cmath>
#include <limits>
 
int hostmult(float * C, float * A, float * B, int Ay, int Ax, int Bx)
{
	// compute C = A * B for matrices (assume Ax = By and C  is Ay x Bx)
	for(int i=0;i<Ay;i++) for(int j=0;j<Bx;j++){
		C[i*Bx+j] = 0.0;      // Cij   = ∑k      Aik  *   Bkj
		for(int k=0;k<Ax;k++) C[i*Bx+j] += A[i*Ax+k]*B[k*Bx+j];
	}
	return 0;
}

bool areFloatEqual(float a, float b, float episilon = std::numeric_limits<float>::epsilon()){
	return std::fabs(a-b) < episilon;
}

bool checkAccuracy(float * C, float * Ref, int size){
	for(int i=0;i<size;i++){
		if(!areFloatEqual(C[i],Ref[i],0.0001f)) return false;
	}
	return true;
}

template <int TS> __global__ void gputiled0(r_Ptr<float> C, cr_Ptr<float> A, cr_Ptr<float> B,int Ay,int Ax,int Bx)
{
	__shared__ float Atile[TS][TS];  // tile in A eg [16][16]
	__shared__ float Btile[TS][TS];  // tile in B eg [16][16]

	int tx  = threadIdx.x;            // tile col index j
	int ty  = threadIdx.y;            // tile row index i
	int ocx = blockDim.x*blockIdx.x;  // tile x origin in C (all threads)    
	int ocy = blockDim.y*blockIdx.y;  // tile y origin in C (all threads)

	int ax = tx;      // j or x in first tile on A
	int ay = ocy+ty;  // i or y in first tile on A and C
	int bx = ocx+tx;  // j or x in first tile on B and C
	int by = ty;      // i or y in first tile on B

	float csum = 0.0f;
#pragma unroll 16
	for(int t=0; t<gridDim.x; t++){
		Atile[ty][tx] = A[ay*Ax+ax];  // copy A tile to shared mem
		Btile[ty][tx] = B[by*Bx+bx];  // copy B tile to shared mem
		__syncthreads();
		for(int k=0;k<TS;k++) csum += Atile[ty][k]*Btile[k][tx];
		__syncthreads();
		ax += TS;         // step A tiles along rows of A
		by += TS;         // step B tiles down  cols of B
	}
	C[ay*Bx+bx] = csum; // store complete result
}

template <int TS, int TM> __global__ void gputiled1(r_Ptr<float> C, cr_Ptr<float> A, cr_Ptr<float> B,int Ay,int Ax,int Bx)
{
	static_assert(TM%4==0&&TM>0,"A bad TM");
	static_assert(TS%TM==0&&TS>TM,"A bad TS");
	// A block still deal with TS*TS elements.
	__shared__ float Atile[TS][TS];  // tile in A eg [16][16]
	__shared__ float Btile[TS][TS];  // tile in B eg [16][16]
       
        // A warp deal with 32*TM elements.	
	float threadResults[TM] = {0.0};
	int tx  = threadIdx.x;            // tile col index j
	int ty  = threadIdx.y;            // tile row index i
	int ocx = blockDim.x * blockIdx.x;  // tile x origin in C (all threads)    
	int ocy = blockDim.y * blockIdx.y * TM;  // tile y origin in C (all threads), tiling along column

	int ax = tx;      // j or x in first tile on A
	int by = ty*TM;   // i or y in first tile on B
	int ay = ocy+by;  // i or y in first tile on A and C
	int bx = ocx+tx;  // j or x in first tile on B and C

#pragma unroll 16
	for(int t=0; t<gridDim.x; t++){
		for(int m=0;m<TM;m++){
		    Atile[ty*TM+m][tx] = A[(ay+m)*Ax+ax];  // copy A tile to shared mem
		    Btile[ty*TM+m][tx] = B[(by+m)*Bx+bx];  // copy B tile to shared mem
		}
		__syncthreads();

		for(int k=0;k<TS;k++) {
		    float Btmp = Btile[k][tx];
		    for (int m=0;m<TM;m++) {
			// deal with TM elements in one column.
	            threadResults[m] += Atile[ty*TM+m][k] * Btmp;
		    }
		}
		__syncthreads();
		ax += TS;         // step A tiles along rows of A
		by += TS;         // step B tiles down  cols of B
	}

#pragma unroll 16
	for(int m=0;m<TM;m++){
	    C[(ay+m)*Bx+bx] = threadResults[m]; // store complete result
	}
}

template <int TS, int TM> __global__ void gputiled2(r_Ptr<float> C, cr_Ptr<float> A, cr_Ptr<float> B,int Ay,int Ax,int Bx)
{
	static_assert(TM%4==0&&TM>0,"A bad TM");
	static_assert(TS%TM==0&&TS>TM,"A bad TS");
	// A block still deals with TS*TS elements.
	__shared__ float Atile[TS][TS];  // tile in A eg [32][32]
	__shared__ float Btile[TS][TS];  // tile in B eg [32][32]
       
	// A warp deals with 32*TM*TM elements.
	// A thread deals with TM*TM elements.
	float threadResults[TM][TM] = {0.0};
	float regA[TM] = {0.0};
	float regB[TM] = {0.0};

	int tx  = threadIdx.x;            // tile col index j
	int ty  = threadIdx.y;            // tile row index i
	int ocx = blockDim.x * blockIdx.x * TM;  // tile x origin in C (all threads), tiling along row
	int ocy = blockDim.y * blockIdx.y * TM;  // tile y origin in C (all threads), tiling along column
	
	// denote that this thread deal with a TM*TM matrix starts from [tx*TM, ty*TM]
	// in an TS*TS (for C) matrix starts from [ocx,ocy].
	int ax = tx*TM;      // j or x in first tile on A
	int by = ty*TM;   // i or y in first tile on B
	int ay = ocy+by;  // i or y in first tile on A and C
	int bx = ocx+ax;  // j or x in first tile on B and C

#pragma unroll 16
	for(int t=0; t<gridDim.x; t++){
		// load TS*TS matrix to smem.
		for(int m=0;m<TM;m++){
		    for(int n=0;n<TM;n++){
		    	Atile[ty*TM+m][tx*TM+n] = A[(ay+m)*Ax+ax+n];  // copy A tile to shared mem
		    	Btile[ty*TM+m][tx*TM+n] = B[(by+m)*Bx+bx+n];  // copy B tile to shared mem
		    }
		}
		__syncthreads();
		
		// the outer(k) loop is like the loop of t.
		for(int k=0;k<TS/TM;k++) {
			// here we deal with a sub-matrix for subA with index starts from [ty*TM,k*TM]
			// and a sub-matrix for subB with index starts from [k*TM,tx*TM]
			// here we split the TM*TM^TM*TM matmul by spliting into vectors mult.
			for(int m=0;m<TM;m++){
				for(int n=0;n<TM;n++){
			    	regA[n] = Atile[ty*TM+n][k*TM+m];
			    	regB[n] = Btile[k*TM+m][tx*TM+n];
				}
				// deal with TM elements in one column.
				for(int i=0;i<TM;i++){
					for(int j=0;j<TM;j++){
						threadResults[i][j] += regA[i] * regB[j];
					}
				}
			}
		}
		__syncthreads();
		ax += TS;         // step A tiles along rows of A
		by += TS;         // step B tiles down  cols of B
	}

#pragma unroll 16
	for(int m=0;m<TM;m++){
		for(int n=0;n<TM;n++){
			C[(ay+m)*Bx+bx+n] = threadResults[m][n]; // store complete result
		}
	}
}

template <int TS, int TM> __global__ void gputiled3(r_Ptr<float> C, cr_Ptr<float> A, cr_Ptr<float> B,int Ay,int Ax,int Bx)
{
	static_assert(TM%4==0&&TM>0,"A bad TM");
	static_assert(TS%TM==0&&TS>TM,"A bad TS");
	// A block still deals with TS*TS elements.
	__shared__ float Atile[TS][TS];  // tile in A eg [32][32]
	__shared__ float Btile[TS][TS];  // tile in B eg [32][32]
       
	// A warp deals with 32*TM*TM elements.
	// A thread deals with TM*TM elements.
	float threadResults[TM][TM] = {0.0};
	float regA[TM] = {0.0};
	float regB[TM] = {0.0};

	int tx  = threadIdx.x;            // tile col index j
	int ty  = threadIdx.y;            // tile row index i
	int ocx = blockDim.x * blockIdx.x * TM;  // tile x origin in C (all threads), tiling along row
	int ocy = blockDim.y * blockIdx.y * TM;  // tile y origin in C (all threads), tiling along column
	
	// denote that this thread deal with a TM*TM matrix starts from [tx*TM, ty*TM]
	// in an TS*TS (for C) matrix starts from [ocx,ocy].
	int ax = tx*TM;      // j or x in first tile on A
	int by = ty*TM;   // i or y in first tile on B
	int ay = ocy+by;  // i or y in first tile on A and C
	int bx = ocx+ax;  // j or x in first tile on B and C

#pragma unroll 16
	for(int t=0; t<gridDim.x; t++){
		// load TS*TS matrix to smem.
		for(int m=0;m<TM;m++){
		    for(int n=0;n<TM;n++){
		    	Atile[tx*TM+n][ty*TM+m] = A[(ay+m)*Ax+ax+n];  // copy A tile to shared mem with transpose
		    	Btile[ty*TM+m][tx*TM+n] = B[(by+m)*Bx+bx+n];  // copy B tile to shared mem
		    }
		}
		__syncthreads();
		
		// the outer(k) loop is like the loop of t.
		for(int k=0;k<TS/TM;k++) {
			// here we deal with a sub-matrix for subA with index starts from [ty*TM,k*TM]
			// and a sub-matrix for subB with index starts from [k*TM,tx*TM]
			// here we split the TM*TM^TM*TM matmul by spliting into vectors mult.
			for(int m=0;m<TM;m++){
				for(int n=0;n<TM;n++){
			    	regA[n] = Atile[k*TM+m][ty*TM+n];
			    	regB[n] = Btile[k*TM+m][tx*TM+n];
				}
				// deal with TM elements in one column.
				for(int i=0;i<TM;i++){
					for(int j=0;j<TM;j++){
						threadResults[i][j] += regA[i] * regB[j];
					}
				}
			}
		}
		__syncthreads();
		ax += TS;         // step A tiles along rows of A
		by += TS;         // step B tiles down  cols of B
	}

#pragma unroll 16
	for(int m=0;m<TM;m++){
		for(int n=0;n<TM;n++){
			C[(ay+m)*Bx+bx+n] = threadResults[m][n]; // store complete result
		}
	}
}




template <int TS, int TM> __global__ void gputiled4(r_Ptr<float> C, cr_Ptr<float> A, cr_Ptr<float> B,int Ay,int Ax,int Bx)
{
	static_assert(TM%4==0&&TM>0,"A bad TM");
	static_assert(TS%TM==0&&TS>TM,"A bad TS");
	// A block still deals with TS*TS elements.
	__shared__ float Atile[TS][TS];  // tile in A eg [32][32]
	__shared__ float Btile[TS][TS];  // tile in B eg [32][32]
       
	// A warp deals with 32*TM*TM elements.
	// A thread deals with TM*TM elements.
	float threadResults[TM][TM] = {0.0};
	float regA[TM] = {0.0};
	float regB[TM] = {0.0};

	int tx  = threadIdx.x;            // tile col index j
	int ty  = threadIdx.y;            // tile row index i
	int ocx = blockDim.x * blockIdx.x * TM;  // tile x origin in C (all threads), tiling along row
	int ocy = blockDim.y * blockIdx.y * TM;  // tile y origin in C (all threads), tiling along column
	
	// denote that this thread deal with a TM*TM matrix starts from [tx*TM, ty*TM]
	// in an TS*TS (for C) matrix starts from [ocx,ocy].
	int ax = tx*TM;      // j or x in first tile on A
	int by = ty*TM;   // i or y in first tile on B
	int ay = ocy+by;  // i or y in first tile on A and C
	int bx = ocx+ax;  // j or x in first tile on B and C

#pragma unroll 16
	for(int t=0; t<gridDim.x; t++){
		// load TS*TS matrix to smem.
		for(int m=0;m<TM;m++){
		    for(int n=0;n<TM;n++){
		    	Atile[tx*TM+n][ty*TM+m] = A[(ay+m)*Ax+ax+n];  // copy A tile to shared mem with transpose
		    }
		}
		for(int m=0;m<TM;m++){
		    for(int n=0;n<TM/4;n++){
		    	reinterpret_cast<float4*>(Btile)[(ty*TM+m)*TS/4+tx*TM/4+n] = reinterpret_cast<const float4*>(B)[(by+m)*Bx/4+bx/4+n]; // copy B tile to shared mem
		    }
		}
		__syncthreads();
		
		// the outer(k) loop is like the loop of t.
		for(int k=0;k<TS/TM;k++) {
			// here we deal with a sub-matrix for subA with index starts from [ty*TM,k*TM]
			// and a sub-matrix for subB with index starts from [k*TM,tx*TM]
			// here we split the TM*TM^TM*TM matmul by spliting into vectors mult.
			for(int m=0;m<TM;m++){
				for(int n=0;n<TM/4;n++){
					reinterpret_cast<float4*>(regA)[n] = reinterpret_cast<float4*>(Atile)[(k*TM+m)*TS/4+ty*TM/4+n];
			    	reinterpret_cast<float4*>(regB)[n] = reinterpret_cast<float4*>(Btile)[(k*TM+m)*TS/4+tx*TM/4+n];
				}
				// deal with TM elements in one column.
				for(int i=0;i<TM;i++){
					for(int j=0;j<TM;j++){
						threadResults[i][j] += regA[i] * regB[j];
					}
				}
			}
		}
		__syncthreads();
		ax += TS;         // step A tiles along rows of A
		by += TS;         // step B tiles down  cols of B
	}

#pragma unroll 16
	for(int m=0;m<TM;m++){
		for(int n=0;n<TM;n++){
			C[(ay+m)*Bx+bx+n] = threadResults[m][n]; // store complete result
		}
	}
}

int main(int argc,char *argv[])
{
	int kernel_index = (argc > 1) ? atoi(argv[1]) : 0;
	int Arow = (argc > 2) ? atoi(argv[2]) : 1 << 10; // default 2^10
	int Acol = (argc > 3) ? atoi(argv[3]) : Arow;
	int Brow = Acol;
	int Bcol = (argc > 4) ? atoi(argv[4]) : Brow;
	int Crow = Arow;
	int Ccol = Bcol;
	uint tilex = (argc > 5) ? atoi(argv[5]) : 32;
	int nacc = (argc > 6) ? atoi(argv[6]) : 100;   // for timing

	thrust::host_vector<float>       A(Arow*Acol);
	thrust::host_vector<float>       B(Brow*Bcol);
	thrust::host_vector<float>       C(Crow*Ccol);
	thrust::host_vector<float>       Ref(Crow*Ccol);
	thrust::device_vector<float> dev_A(Arow*Acol);
	thrust::device_vector<float> dev_B(Brow*Bcol);
	thrust::device_vector<float> dev_C(Crow*Ccol);

	// initialise x with random numbers and copy to dx.
	std::default_random_engine gen(12345678);
	std::uniform_real_distribution<float> fran(0.0,1.0);
	for(int k = 0; k<Arow*Acol; k++) A[k] = fran(gen);
	for(int k = 0; k<Brow*Bcol; k++) B[k] = fran(gen);
	hostmult(Ref.data(),A.data(),B.data(),Arow,Acol,Bcol);

	dev_A = A;  // H2D copy
	dev_B = B;  // H2D copy
	
	double t3 = 0.0;
	if(kernel_index == 0){
		dim3 threads ={tilex,tilex,1}; // force square
		dim3 blocks ={(Bcol+threads.x-1)/threads.x,(Arow+threads.y-1)/threads.y,1};
	
		cx::timer tim;
		for(int k=0;k<nacc;k++){
			if(tilex == 8)	     gputiled0< 8><<<blocks,threads>>>(dev_C.data().get(),dev_A.data().get(),dev_B.data().get(),Arow,Acol,Bcol);
			else if(tilex == 16) gputiled0<16><<<blocks,threads>>>(dev_C.data().get(),dev_A.data().get(),dev_B.data().get(),Arow,Acol,Bcol);
			else if(tilex == 32) gputiled0<32><<<blocks,threads>>>(dev_C.data().get(),dev_A.data().get(),dev_B.data().get(),Arow,Acol,Bcol);
		}
		cudaDeviceSynchronize();
		t3 = tim.lap_ms()/(double)(nacc);
	} else if(kernel_index == 1){
		constexpr int TM = 8;
		dim3 threads = {tilex, tilex/TM, 1}; //force square
		dim3 blocks =  {(Bcol+threads.x-1)/threads.x,(Arow+threads.x-1)/threads.x,1};

		cx::timer tim;
		for(int k=0;k<nacc;k++){
			if(tilex == 32) gputiled1<32,TM><<<blocks,threads>>>(dev_C.data().get(),dev_A.data().get(),dev_B.data().get(),Arow,Acol,Bcol);
		}
		cudaDeviceSynchronize();
		t3 = tim.lap_ms()/(double)(nacc);
	} else if(kernel_index == 2){
		constexpr int TM = 4;
		dim3 threads = {tilex/TM, tilex/TM, 1}; //force square
		dim3 blocks =  {(Bcol+threads.x-1)/tilex,(Arow+threads.x-1)/tilex,1};

		cx::timer tim;
		for(int k=0;k<nacc;k++){
			if(tilex == 32) gputiled2<32,TM><<<blocks,threads>>>(dev_C.data().get(),dev_A.data().get(),dev_B.data().get(),Arow,Acol,Bcol);
		}
		cudaDeviceSynchronize();
		t3 = tim.lap_ms()/(double)(nacc);
	} else if(kernel_index == 3){
		constexpr int TM = 4;
		dim3 threads = {tilex/TM, tilex/TM, 1}; //force square
		dim3 blocks =  {(Bcol+threads.x-1)/tilex,(Arow+threads.x-1)/tilex,1};

		cx::timer tim;
		for(int k=0;k<nacc;k++){
			if(tilex == 32) gputiled3<32,TM><<<blocks,threads>>>(dev_C.data().get(),dev_A.data().get(),dev_B.data().get(),Arow,Acol,Bcol);
		}
		cudaDeviceSynchronize();
		t3 = tim.lap_ms()/(double)(nacc);
	} else if(kernel_index == 4){
		constexpr int TM = 4;
		dim3 threads = {tilex/TM, tilex/TM, 1}; //force square
		dim3 blocks =  {(Bcol+threads.x-1)/tilex,(Arow+threads.x-1)/tilex,1};

		cx::timer tim;
		for(int k=0;k<nacc;k++){
			if(tilex == 32) gputiled4<32,TM><<<blocks,threads>>>(dev_C.data().get(),dev_A.data().get(),dev_B.data().get(),Arow,Acol,Bcol);
		}
		cudaDeviceSynchronize();
		t3 = tim.lap_ms()/(double)(nacc);
	} else {
		std::cout << "Unsupported kernel!" << std::endl;
	}
	C = dev_C; // D2H copy

	double flops = 2.0*(double)Arow*(double)Acol*(double)Bcol;
	double gflops = flops/(t3*1000000.0);
	double gbytes = gflops*6.0; // i.e 12 bytes per term
	bool accuracy = checkAccuracy(C.data(), Ref.data(), Crow*Ccol);
	printf("A %d x %d B %d x %d, accuracy %s, gpu time %.3f ms, GFlops %.3f, GBytes %.3f (gputiled)\n",Arow,Acol,Brow,Bcol,accuracy?"true":"false",t3,gflops,gbytes);

	return 0;
}
