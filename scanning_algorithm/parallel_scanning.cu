/*
    EE 451 Final Project
    Scanning Algorithm
    PARALLEL VERSION
    NOTE: NOT FULLY IMPLEMENTED YET, OUTPUTS ARE NOT FULLY CORRECT
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <cublas.h>

#define h  800
#define w  800

#define input_file  "images/input_800x800.raw"
#define output_file "output.raw"

#define VMAX 512
#define LMAX 255

// global variables
unsigned char *f; 
unsigned char *v;
unsigned char *l;
// int lMin;
// int fMin;
int modified;
int currentLabel;

// if f[p] has lower neighbors, v[p] becomes 1, otherwise 0
void step1(int r, int c) {
    // converting r, c pair to index in v
    int p = r * w + c;
    if (v[p] != 1) {
        // for each neighbor, if f[n_p] < f[p], v[p] = 1
        // accessing neighbors: top left (a), above (b), top right (c), right (d), bottom right (e), bottom (f), bottom left (g), left (h)
        // -------------------------
        // |       |       |       |
        // |   a   |   b   |   c   |
        // |       |       |       |
        // -------------------------
        // |       |       |       |
        // |   h   |       |   d   |
        // |       |       |       |
        // -------------------------
        // |       |       |       |
        // |   g   |   f   |   e   |
        // |       |       |       |
        // -------------------------

        if (r > 0) {
            // above: b
            if (f[(r-1)*w + c] <  f[r * w + c]) {
                v[p] = 1;
                return;
            }
            if (c > 0) {
                // top left: a
                if (f[(r-1)*w + c-1] <  f[r * w + c]) {
                    v[p] = 1;
                    return;
                }
            }
            if (c < w - 1) {
                // top right: c
                if (f[(r-1)*w + c+1] <  f[r * w + c]) {
                    v[p] = 1;
                    return;
                }
            }
        }
        if (r < h - 1) {
            // below: f
            if (f[(r+1)*w + c] <  f[r * w + c]) {
                v[p] = 1;
                return;
            }
            if (c > 0) {
                // bottom left: g
                if (f[(r+1)*w + c-1] <  f[r * w + c]) {
                    v[p] = 1;
                    return;
                }
            }
            if (c < w - 1) {
                // bottom right: e
                if (f[(r+1)*w + c+1] <  f[r * w + c]) {
                    v[p] = 1;
                    return;
                }
            }
        }
        if (c > 0) {
            // left: h
            if (f[r*w + c-1] <  f[r * w + c]) {
                v[p] = 1;
                return;
            }
        }
        if (c < w - 1) {
            // right: d
            if (f[r*w + c+1] <  f[r * w + c]) {
                v[p] = 1;
                return;
            }
        }

    }
}

// at local minima, values of v[p] become 0, otherwise shortest distance from lowest adjacent plateau
__device__ int step2(int r, int c, char *f, char *v) {
    // converting r, c pair to index in v
    int p = r * w + c;
    int min;
    int mod = 0;

    if (v[p] != 1) {
        // for each neighbor with the same gray level v[n_p] > 0, update min if needed
        min = VMAX;
        int np;

        // looping through neighbors
        if (r > 0) {
            // above: b
            np = (r-1) * w + c; // neighbor p
            if (f[(r-1)*w + c] ==  f[r * w + c] && v[np] > 0) {
                if (v[np] < min) min = v[np];
            }
            if (c > 0) {
                // top left: a
                np = (r-1) * w + (c-1);
                if (f[(r-1)*w + c-1] ==  f[r * w + c] && v[np] > 0) {
                    if (v[np] < min) min = v[np];
                }
            }
            if (c < w - 1) {
                // top right: c
                np = (r-1) * w + (c+1);
                if (f[(r-1)*w + c+1] ==  f[r * w + c] && v[np] > 0) {
                    if (v[np] < min) min = v[np];
                }
            }
        }
        if (r < h - 1) {
            // below: f
            np = (r+1) * w + c; // neighbor p
            if (f[(r+1)*w + c] ==  f[r * w + c] && v[np] > 0) {
                if (v[np] < min) min = v[np];
            }
            if (c > 0) {
                // bottom left: g
                np = (r+1) * w + (c-1);
                if (f[(r+1)*w + c-1] ==  f[r * w + c] && v[np] > 0) {
                   if (v[np] < min) min = v[np]; 
                }
            }
            if (c < w - 1) {
                // bottom right: e
                np = (r+1) * w + (c+1);
                if (f[(r+1)*w + c+1] ==  f[r * w + c] && v[np] > 0) {
                    if (v[np] < min) min = v[np];
                }
            }
        }
        if (c > 0) {
            // left: h
            np = r * w + (c-1);
            if (f[r*w + c-1] ==  f[r * w + c] && v[np] > 0) {
                if (v[np] < min) min = v[np]; 
            }
        }
        if (c < w - 1) {
            // right: d
            np = r * w + (c+1);
            if (f[r*w + c+1] ==  f[r * w + c] && v[np] > 0) {
                if (v[np] < min) min = v[np]; 
            }
        }

        // updating v[p] if necessary
        if (min != VMAX && v[p] != (min + 1)) {
            v[p] = min + 1;
            mod = 1;
        }

    }
    return mod;
}

__device__ int step3(int r, int c, char *f, char *v, char *l, int currentLabel) {
    int p = r * w + c;
    int np, lMin, fMin;

    int mod = 0;

    // minimal plateaus
    if (v[p] == 0) {
        lMin = LMAX;
        // for each neighbor, update lmin if needed
        if (r > 0) {
            // above: b
            np = (r-1) * w + c; // neighbor p
            if (f[(r-1)*w + c] ==  f[r * w + c]) {
                if (l[np] > 0 && l[np] < lMin) lMin = l[np];
            }
            if (c > 0) {
                // top left: a
                np = (r-1) * w + (c-1);
                if (f[(r-1)*w + c-1] ==  f[r * w + c]) {
                    if (l[np] > 0 && l[np] < lMin) lMin = l[np];
                }
            }
            if (c < w - 1) {
                // top right: c
                np = (r-1) * w + (c+1);
                if (f[(r-1)*w + c+1] ==  f[r * w + c]) {
                    if (l[np] > 0 && l[np] < lMin) lMin = l[np];
                }
            }
        }
        if (r < h - 1) {
            // below: f
            np = (r+1) * w + c; // neighbor p
            if (f[(r+1)*w + c] ==  f[r * w + c]) {
                if (l[np] > 0 && l[np] < lMin) lMin = l[np];
            }
            if (c > 0) {
                // bottom left: g
                np = (r+1) * w + (c-1);
                if (f[(r+1)*w + c-1] ==  f[r * w + c]) {
                   if (l[np] > 0 && l[np] < lMin) lMin = l[np];
                }
            }
            if (c < w - 1) {
                // bottom right: e
                np = (r+1) * w + (c+1);
                if (f[(r+1)*w + c+1] ==  f[r * w + c]) {
                    if (l[np] > 0 && l[np] < lMin) lMin = l[np];
                }
            }
        }
        if (c > 0) {
            // left: h
            np = r * w + (c-1);
            if (f[r*w + c-1] ==  f[r * w + c]) {
                if (l[np] > 0 && l[np] < lMin) lMin = l[np];
            }
        }
        if (c < w - 1) {
            // right: d
            np = r * w + (c+1);
            if (f[r*w + c+1] ==  f[r * w + c]) {
                if (l[np] > 0 && l[np] < lMin) lMin = l[np];
            }
        }

        if (lMin == LMAX && l[p] == 0) {
            lMin = currentLabel++;
        }

    }

    // edges of non-minimal plateaus
    else if (v[p] == 1) {
        fMin =  f[r * w + c];

        // looping through neighbors
        if (r > 0) {
            // above: b
            if (f[(r-1)*w + c] < fMin) fMin = f[(r-1)*w + c];
            if (c > 0) {
                // top left: a
                if (f[(r-1)*w + c-1] < fMin) fMin = f[(r-1)*w + c-1];
            }
            if (c < w - 1) {
                // top right: c
                if (f[(r-1)*w + c+1] < fMin) fMin = f[(r-1)*w + c+1];
            }
        }
        if (r < h - 1) {
            // below: f
            if (f[(r+1)*w + c] < fMin) fMin = f[(r+1)*w + c];
            if (c > 0) {
                // bottom left: g
                if (f[(r+1)*w + c-1] < fMin) fMin = f[(r+1)*w + c-1];
            }
            if (c < w - 1) {
                // bottom right: e
                if (f[(r+1)*w + c+1] < fMin) fMin = f[(r+1)*w + c+1];
            }
        }
        if (c > 0) {
            // left: h
            if (f[r*w + c-1] < fMin) fMin = f[r*w + c-1];
        }
        if (c < w - 1) {
            // right: d
            if (f[r*w + c+1] < fMin) fMin = f[r*w + c+1];
        }


        lMin = LMAX;
        if (r > 0) {
            // above: b
            np = (r-1) * w + c; // neighbor p
            if (f[(r-1)*w + c] == fMin) {
                if (l[np] > 0 && l[np] < lMin) lMin = l[np];
            }
            if (c > 0) {
                // top left: a
                np = (r-1) * w + (c-1);
                if (f[(r-1)*w + c-1] == fMin) {
                    if (l[np] > 0 && l[np] < lMin) lMin = l[np];
                }
            }
            if (c < w - 1) {
                // top right: c
                np = (r-1) * w + (c+1);
                if (f[(r-1)*w + c+1] == fMin) {
                    if (l[np] > 0 && l[np] < lMin) lMin = l[np];
                }
            }
        }
        if (r < h - 1) {
            // below: f
            np = (r+1) * w + c; // neighbor p
            if (f[(r+1)*w + c] == fMin) {
                if (l[np] > 0 && l[np] < lMin) lMin = l[np];
            }
            if (c > 0) {
                // bottom left: g
                np = (r+1) * w + (c-1);
                if (f[(r+1)*w + c-1] == fMin) {
                   if (l[np] > 0 && l[np] < lMin) lMin = l[np];
                }
            }
            if (c < w - 1) {
                // bottom right: e
                np = (r+1) * w + (c+1);
                if (f[(r+1)*w + c+1] == fMin) {
                    if (l[np] > 0 && l[np] < lMin) lMin = l[np];
                }
            }
        }
        if (c > 0) {
            // left: h
            np = r * w + (c-1);
            if (f[r*w + c-1] == fMin) {
                if (l[np] > 0 && l[np] < lMin) lMin = l[np];
            }
        }
        if (c < w - 1) {
            // right: d
            np = r * w + (c+1);
            if (f[r*w + c+1] == fMin) {
                if (l[np] > 0 && l[np] < lMin) lMin = l[np];
            }
        }


    }

    // non-minimal plateaus
    else {
        lMin = LMAX;
        if (r > 0) {
            // above: b
            np = (r-1) * w + c; // neighbor p
            if (f[(r-1)*w + c] ==  f[r * w + c]) {
                if (v[np] == v[p]-1 && l[np] > 0 && l[np] < lMin) lMin = l[np];
            }
            if (c > 0) {
                // top left: a
                np = (r-1) * w + (c-1);
                if (f[(r-1)*w + c-1] ==  f[r * w + c]) {
                    if (v[np] == v[p]-1 && l[np] > 0 && l[np] < lMin) lMin = l[np];
                }
            }
            if (c < w - 1) {
                // top right: c
                np = (r-1) * w + (c+1);
                if (f[(r-1)*w + c+1] ==  f[r * w + c]) {
                    if (v[np] == v[p]-1 && l[np] > 0 && l[np] < lMin) lMin = l[np];
                }
            }
        }
        if (r < h - 1) {
            // below: f
            np = (r+1) * w + c; // neighbor p
            if (f[(r+1)*w + c] ==  f[r * w + c]) {
                if (v[np] == v[p]-1 && l[np] > 0 && l[np] < lMin) lMin = l[np];
            }
            if (c > 0) {
                // bottom left: g
                np = (r+1) * w + (c-1);
                if (f[(r+1)*w + c-1] ==  f[r * w + c]) {
                   if (v[np] == v[p]-1 && l[np] > 0 && l[np] < lMin) lMin = l[np];
                }
            }
            if (c < w - 1) {
                // bottom right: e
                np = (r+1) * w + (c+1);
                if (f[(r+1)*w + c+1] ==  f[r * w + c]) {
                    if (v[np] == v[p]-1 && l[np] > 0 && l[np] < lMin) lMin = l[np];
                }
            }
        }
        if (c > 0) {
            // left: h
            np = r * w + (c-1);
            if (f[r*w + c-1] ==  f[r * w + c]) {
                if (v[np] == v[p]-1 && l[np] > 0 && l[np] < lMin) lMin = l[np];
            }
        }
        if (c < w - 1) {
            // right: d
            np = r * w + (c+1);
            if (f[r*w + c+1] ==  f[r * w + c]) {
                if (v[np] == v[p]-1 && l[np] > 0 && l[np] < lMin) lMin = l[np];
            }
        }

    }

    // setting label for this pixel
    if (lMin != LMAX && l[p] != lMin) {
        l[p] = lMin;
        mod = 1;
    }

    return mod;
}


// KERNEL FUNCTIONS
__global__ void step2loop(char *f, char *v) {

    // determining which thread in the overall grid we are
    int my_x = blockIdx.x*blockDim.x + threadIdx.x;
    int my_y = blockIdx.y*blockDim.y + threadIdx.y;

    int mod;

    while (1) {
        mod = 0;
        // scan from top-left to bottom-right(p)
            // step2(p)

        for (int row = my_x; row < my_x + 16; row++) {
            for (int col = my_y; col < my_y + 16; col++) {
                mod = step2(row, col, f, v) || mod;
            }
        }

        __syncthreads();

        // if v[p] is not modified
            // break
        if (!mod) {
            break;
        }

        // scan from bottom-right to top-left(p)
            // step2(p)
        mod = 0;
        for (int row = my_x + 15; row >= my_x; row--) {
            for (int col = my_y + 15; col >= my_y; col--) {
                mod = step2(row, col, f, v) || mod;
            }
        }

        __syncthreads();
        
        // if v[p] is not modified
            // break
        if (!mod) {
            break;
        }

    }

}

__global__ void step3loop(char *f, char *v, char *l) {

    // determining which thread in the overall grid we are
    int my_x = blockIdx.x*blockDim.x + threadIdx.x;
    int my_y = blockIdx.y*blockDim.y + threadIdx.y;

    int mod;
    int currentLabel = 1;

    while (1) {
        mod = 0;
        // scan from top-left to bottom-right(p)
            // step2(p)

        for (int row = my_x; row < my_x + 16; row++) {
            for (int col = my_y; col < my_y + 16; col++) {
                mod = step3(row, col, f, v, l, currentLabel) || mod;
            }
        }

        __syncthreads();

        // if v[p] is not modified
            // break
        if (!mod) {
            break;
        }

        // scan from bottom-right to top-left(p)
            // step2(p)
        mod = 0;
        for (int row = my_x + 15; row >= my_x; row--) {
            for (int col = my_y + 15; col >= my_y; col--) {
                mod = step3(row, col, f, v, l, currentLabel) || mod;
            }
        }

        __syncthreads();
        
        // if v[p] is not modified
            // break
        if (!mod) {
            break;
        }

    }

}


int main(int argc, char** argv) {
    printf("in main\n");

    struct timespec start, stop; 
	double time;

    FILE *fp;
    f = (unsigned char*) malloc (sizeof(unsigned char)*h*w);
    // the matrix is stored in a linear array in row major fashion
	if (!(fp=fopen(input_file, "rb"))) {
		printf("can not open file\n");
		return 1;
	}
	fread(f, sizeof(unsigned char), w*h, fp);
	fclose(fp);

     printf("got file\n");

    // ALGORITHM IMPLEMENTATION BEGINS ----------------------------------------------

    // INITIALIZATION 
    // setting v[p] and l[p] values matrix to 0s
    v = (unsigned char*) malloc (sizeof(unsigned char)*h*w);
    l = (unsigned char*) malloc (sizeof(unsigned char)*h*w);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            v[i * w + j] = 0;
            l[i * w + j] = 0;
        }
    }
    
    // STARTING TIME
    if( clock_gettime(CLOCK_REALTIME, &start) == -1) { perror("clock gettime");}

    // step 1 (serial)
    for (int row = 0; row < h; row++) {
        for (int col = 0; col < w; col++) {
            step1(row, col);
        }
    }

    // dimensions are 800 x 800
    dim3 dimGrid(50, 50);
    dim3 dimBlock(16, 16);

    // allocating space for f, v, and l matrices on GPU
    char *gpu_f,  *gpu_v, *gpu_l;
    cudaMalloc((void**) &gpu_f, sizeof(char)*w*h);
    cudaMalloc((void**) &gpu_v, sizeof(char)*w*h);
    cudaMalloc((void**) &gpu_l, sizeof(char)*w*h);

    // copying over initial values of f, v, and l from CPU to GPU
    cudaMemcpy(gpu_f, f, sizeof(char)*w*h, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_v, v, sizeof(char)*w*h, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_l, l, sizeof(char)*w*h, cudaMemcpyHostToDevice);


    // launch kernel function for step2loop
    printf("\nabout to launch step2loop kernel\n");
    step2loop<<<dimGrid, dimBlock>>>(gpu_f, gpu_v);
    cudaMemcpy(v, gpu_v, sizeof(char)*w*h, cudaMemcpyDeviceToHost); // copying updated v back to host

    // launch kernel function for step3loop
    printf("\nabout to launch step3loop kernel\n");
    step3loop<<<dimGrid, dimBlock>>>(gpu_f, gpu_v, gpu_l);
    cudaMemcpy(l, gpu_l, sizeof(char)*w*h, cudaMemcpyDeviceToHost); // copying updated v back to host


    // STOPPING TIME
    if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror("clock gettime");}		
	time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e6;
    printf("Parallel execution time = %f ms\n", time);	

    if (!(fp=fopen(output_file,"wb"))) {
		printf("can not open file\n");
		return 1;
	}	
	fwrite(l, sizeof(unsigned char),w*h, fp);
    fclose(fp);

    free(f);
    free(v);
    free(l);
    cudaFree(gpu_f);
    cudaFree(gpu_v);
    cudaFree(gpu_l);

    return 0;
}