#include <stdlib.h>
#include <stdio.h>
#include <cublas.h>
#include <time.h>

#define height  800 
#define width  800

#define input_file  "input.raw"
#define output_file "output.raw"
#define INIT -1
#define WATERSHED 0

const int blockH = 32;
const int blockW = 32;
int blockPCol = height / blockH;
int blockPRow = width / blockW;
//returns -1 if none of its neighbors are labelled
//returns label if some neighbors are labelled and labelled the same
//returns 0 if neighbors are labelled and labelled differently

__device__ char checkNeighbors(int row, int j, int* a){
    int neighbors[8] = {0,0,0,0,0,0,0,0};
    //   0,1,2
    //   3, ,4
    //   5,6,7
    if(row == 0){
        neighbors[0] = neighbors[1] = neighbors[2] = -2;
    }
    if(row== width-1){
        neighbors[5] = neighbors[6] = neighbors[7] = -2;
    }
    if(j == 0){
        neighbors[0] = neighbors[3] = neighbors[5] = -2;
    }
    else if (j == height-1){
        neighbors[2] = neighbors[4] = neighbors[7] = -2;
    }

    //check if neighbors are labelled
    int temp = -1;
    for(int i = 0; i < 8; i++){
        if(neighbors[i] != -2){
            switch(i){
            case 0:
                neighbors[i] = a[(row-1) * width + j-1];
                break;
            case 1:
                neighbors[i] = a[(row-1) * width + j];
                break;
            case 2:
                neighbors[i] = a[(row-1) * width + j+1];
                break;
            case 3:
                neighbors[i] = a[(row) * width + j-1];
                break;
            case 4:
                neighbors[i] = a[(row) * width + j+1];
                break;
            case 5:
                neighbors[i] = a[(row+1) * width + j-1];
                break;
            case 6:
                neighbors[i] = a[(row+1) * width + j];
                break;
            case 7:
                neighbors[i] = a[(row+1) * width + j+1];
                break;
            }
            if(neighbors[i] != -1){
                if(temp == -1){
                    temp = neighbors[i];
                }
                else if(temp != neighbors[i]){
                    return 0;
                }
            }
        }
    }
    if(temp != -1){
        return temp;
    }
    return -1;
}

__global__ void multiply(unsigned char *a, int *labels){
	int my_x, my_y;
	my_x = blockIdx.x * blockH + threadIdx.x;
	my_y = blockIdx.y * blockW + threadIdx.y;

    int nextLabel = 1;
    for(int i = 0; i < blockH; i++){
        for(int j = 0; j < blockW; j++){
            labels[(my_x + i) * width + my_y + j] = -1;
        }
    }
    __syncthreads();

    //printf("x %d y %d \n", my_x, my_y);
	for(int alt = 0; alt < 256; alt++){
        //start loop 
        for(int i = 0; i < blockH; i++){
            for(int j = 0; j < blockW; j++){
                if(a[(my_x + i) * width + my_y + j] == alt){
                    int temp = checkNeighbors(my_x + i, my_y + j, labels);
                    //no labelled neighbors
                    if(temp == -1){
                        labels[(my_x + i) * width + my_y + j] = ++nextLabel;
                    }
                    //diff labelled neighbors
                    else if(temp == 0){
                        labels[(my_x + i) * width + my_y + j] = WATERSHED;
                    }
                    //same labelled neighbors
                    else{
                        labels[(my_x + i) * width + my_y + j] = temp;
                    }
                }
            }
        }
		__syncthreads();
    }

    for(int i = 0; i < blockH; i++){
        for(int j = 0; j < blockW; j++){
            if(labels[(my_x + i) * width + my_y + j] != WATERSHED){
                //printf("not water shed\n");
                labels[(my_x + i) * width + my_y + j] = 255;
            }
        }
    }
}

int main(){		
	FILE *fp;
	struct timespec start, stop; 
	double time;

  	unsigned char *a = (unsigned char*) malloc (sizeof(unsigned char)*height*width);
  	int *labels = (int*) malloc (sizeof(int)*height*width);
    for(int i = 0; i < height * width ; i++){
        labels[i] = 255;
    }
	// the matrix is stored in a linear array in row major fashion
	if (!(fp=fopen(input_file, "rb"))) {
		printf("can not opern file\n");
		return 1;
	}
	fread(a, sizeof(unsigned char), width*height, fp);
	fclose(fp);
          	

    unsigned char *gpu_a;
    int *gpu_b;
    cudaMalloc((void**)&gpu_a, sizeof(unsigned char)*width*height); 
    cudaMalloc((void**)&gpu_b, sizeof(int)*width*height);	  
    
    cudaMemcpy(gpu_a, a, sizeof(unsigned char)*width*height, cudaMemcpyHostToDevice);
    
    dim3 dimGrid(blockPCol, blockPRow);
    dim3 dimBlock(1, 1);
    
    if( clock_gettime( CLOCK_REALTIME, &start) == -1 ) { perror( "clock gettime" );}
    multiply<<<dimGrid, dimBlock>>>(gpu_a, gpu_b);				
    cudaMemcpy(labels, gpu_b, sizeof(int)*width*height, cudaMemcpyDeviceToHost);
    
    if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror( "clock gettime" );}	  
    time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
    printf("time is %f ns\n", time*1e9);	 
    
    if (!(fp=fopen(output_file,"wb"))) {
        printf("can not opern file\n");
        return 1;
    }	

    unsigned char *b = (unsigned char*) malloc (sizeof(unsigned char)*height*width);
    for(int i = 0; i < height * width ; i++){
        b[i] = labels[i];
    }
    fwrite(b, sizeof(unsigned char),width*height, fp);
    fclose(fp);
    
    free(a);
    free(b);
    free(labels);
    cudaFree(gpu_a);  
    cudaFree(gpu_b);  
    return 0;
}	
