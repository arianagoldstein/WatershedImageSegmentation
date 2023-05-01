/*
    EE 451 Final Project
    Scanning Algorithm
    SERIAL VERSION WITH BASIC MATRIX INPUT
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define h  11
#define w  10

#define VMAX 15
#define LMAX 15

// input array
int f[h][w] = {
    {3,  5,  5,  2,  8,  8,  8,  11, 10, 10},
    {5,  5,  11, 11, 8,  11, 11, 8,  10, 10},
    {11, 5,  11, 11, 9,  9,  9,  9,  8,  10},
    {11, 11, 11, 7,  7,  7,  7,  9,  9,  8},
    {11, 11, 11, 11, 11, 9,  7,  10, 8,  10},
    {11, 10, 11, 9,  7,  7,  9,  9,  10,  8},
    {11, 10, 11, 9,  11, 9,  10, 10, 8,  10},
    {11, 11, 11, 8,  8,  8,  8,  8,  10, 10},
    {11, 11, 11, 11, 10, 10, 10, 10, 10, 10},
    {10, 10, 10, 10, 10, 10, 10, 10, 10, 10},
    {11, 11, 11, 11, 10, 10, 10, 10, 10, 10}
};

// global variables
unsigned char *v;
unsigned char *l;
int lMin;
int fMin;
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
            if (f[r-1][c] < f[r][c]) {
                v[p] = 1;
                return;
            }
            if (c > 0) {
                // top left: a
                if (f[r-1][c-1] < f[r][c]) {
                    v[p] = 1;
                    return;
                }
            }
            if (c < w - 1) {
                // top right: c
                if (f[r-1][c+1] < f[r][c]) {
                    v[p] = 1;
                    return;
                }
            }
        }
        if (r < h - 1) {
            // below: f
            if (f[r+1][c] < f[r][c]) {
                v[p] = 1;
                return;
            }
            if (c > 0) {
                // bottom left: g
                if (f[r+1][c-1] < f[r][c]) {
                    v[p] = 1;
                    return;
                }
            }
            if (c < w - 1) {
                // bottom right: e
                if (f[r+1][c+1] < f[r][c]) {
                    v[p] = 1;
                    return;
                }
            }
        }
        if (c > 0) {
            // left: h
            if (f[r][c-1] < f[r][c]) {
                v[p] = 1;
                return;
            }
        }
        if (c < w - 1) {
            // right: d
            if (f[r][c+1] < f[r][c]) {
                v[p] = 1;
                return;
            }
        }

    }
}

// at local minima, values of v[p] become 0, otherwise shortest distance from lowest adjacent plateau
void step2(int r, int c) {
    // converting r, c pair to index in v
    int p = r * w + c;
    int min;

    if (v[p] != 1) {
        // for each neighbor with the same gray level v[n_p] > 0, update min if needed
        min = VMAX;
        int np;

        // looping through neighbors
        if (r > 0) {
            // above: b
            np = (r-1) * w + c; // neighbor p
            if (f[r-1][c] == f[r][c] && v[np] > 0) {
                if (v[np] < min) min = v[np];
            }
            if (c > 0) {
                // top left: a
                np = (r-1) * w + (c-1);
                if (f[r-1][c-1] == f[r][c] && v[np] > 0) {
                    if (v[np] < min) min = v[np];
                }
            }
            if (c < w - 1) {
                // top right: c
                np = (r-1) * w + (c+1);
                if (f[r-1][c+1] == f[r][c] && v[np] > 0) {
                    if (v[np] < min) min = v[np];
                }
            }
        }
        if (r < h - 1) {
            // below: f
            np = (r+1) * w + c; // neighbor p
            if (f[r+1][c] == f[r][c] && v[np] > 0) {
                if (v[np] < min) min = v[np];
            }
            if (c > 0) {
                // bottom left: g
                np = (r+1) * w + (c-1);
                if (f[r+1][c-1] == f[r][c] && v[np] > 0) {
                   if (v[np] < min) min = v[np]; 
                }
            }
            if (c < w - 1) {
                // bottom right: e
                np = (r+1) * w + (c+1);
                if (f[r+1][c+1] == f[r][c] && v[np] > 0) {
                    if (v[np] < min) min = v[np];
                }
            }
        }
        if (c > 0) {
            // left: h
            np = r * w + (c-1);
            if (f[r][c-1] == f[r][c] && v[np] > 0) {
                if (v[np] < min) min = v[np]; 
            }
        }
        if (c < w - 1) {
            // right: d
            np = r * w + (c+1);
            if (f[r][c+1] == f[r][c] && v[np] > 0) {
                if (v[np] < min) min = v[np]; 
            }
        }

        // updating v[p] if necessary
        if (min != VMAX && v[p] != (min + 1)) {
            v[p] = min + 1;
            modified = 1;
        }

    }
}

void step3(int r, int c) {
    int p = r * w + c;
    int np;

    // minimal plateaus
    if (v[p] == 0) {
        lMin = LMAX;
        // for each neighbor, update lmin if needed
        if (r > 0) {
            // above: b
            np = (r-1) * w + c; // neighbor p
            if (f[r-1][c] == f[r][c]) {
                if (l[np] > 0 && l[np] < lMin) lMin = l[np];
            }
            if (c > 0) {
                // top left: a
                np = (r-1) * w + (c-1);
                if (f[r-1][c-1] == f[r][c]) {
                    if (l[np] > 0 && l[np] < lMin) lMin = l[np];
                }
            }
            if (c < w - 1) {
                // top right: c
                np = (r-1) * w + (c+1);
                if (f[r-1][c+1] == f[r][c]) {
                    if (l[np] > 0 && l[np] < lMin) lMin = l[np];
                }
            }
        }
        if (r < h - 1) {
            // below: f
            np = (r+1) * w + c; // neighbor p
            if (f[r+1][c] == f[r][c]) {
                if (l[np] > 0 && l[np] < lMin) lMin = l[np];
            }
            if (c > 0) {
                // bottom left: g
                np = (r+1) * w + (c-1);
                if (f[r+1][c-1] == f[r][c]) {
                   if (l[np] > 0 && l[np] < lMin) lMin = l[np];
                }
            }
            if (c < w - 1) {
                // bottom right: e
                np = (r+1) * w + (c+1);
                if (f[r+1][c+1] == f[r][c]) {
                    if (l[np] > 0 && l[np] < lMin) lMin = l[np];
                }
            }
        }
        if (c > 0) {
            // left: h
            np = r * w + (c-1);
            if (f[r][c-1] == f[r][c]) {
                if (l[np] > 0 && l[np] < lMin) lMin = l[np];
            }
        }
        if (c < w - 1) {
            // right: d
            np = r * w + (c+1);
            if (f[r][c+1] == f[r][c]) {
                if (l[np] > 0 && l[np] < lMin) lMin = l[np];
            }
        }

        if (lMin == LMAX && l[p] == 0) {
            lMin = currentLabel++;
        }

    }

    // edges of non-minimal plateaus
    else if (v[p] == 1) {
        fMin = f[r][c];

        // looping through neighbors
        if (r > 0) {
            // above: b
            if (f[r-1][c] < fMin) fMin = f[r-1][c];
            if (c > 0) {
                // top left: a
                if (f[r-1][c-1] < fMin) fMin = f[r-1][c-1];
            }
            if (c < w - 1) {
                // top right: c
                if (f[r-1][c+1] < fMin) fMin = f[r-1][c+1];
            }
        }
        if (r < h - 1) {
            // below: f
            if (f[r+1][c] < fMin) fMin = f[r+1][c];
            if (c > 0) {
                // bottom left: g
                if (f[r+1][c-1] < fMin) fMin = f[r+1][c-1];
            }
            if (c < w - 1) {
                // bottom right: e
                if (f[r+1][c+1] < fMin) fMin = f[r+1][c+1];
            }
        }
        if (c > 0) {
            // left: h
            if (f[r][c-1] < fMin) fMin = f[r][c-1];
        }
        if (c < w - 1) {
            // right: d
            if (f[r][c+1] < fMin) fMin = f[r][c+1];
        }


        lMin = LMAX;
        if (r > 0) {
            // above: b
            np = (r-1) * w + c; // neighbor p
            if (f[r-1][c] == fMin) {
                if (l[np] > 0 && l[np] < lMin) lMin = l[np];
            }
            if (c > 0) {
                // top left: a
                np = (r-1) * w + (c-1);
                if (f[r-1][c-1] == fMin) {
                    if (l[np] > 0 && l[np] < lMin) lMin = l[np];
                }
            }
            if (c < w - 1) {
                // top right: c
                np = (r-1) * w + (c+1);
                if (f[r-1][c+1] == fMin) {
                    if (l[np] > 0 && l[np] < lMin) lMin = l[np];
                }
            }
        }
        if (r < h - 1) {
            // below: f
            np = (r+1) * w + c; // neighbor p
            if (f[r+1][c] == fMin) {
                if (l[np] > 0 && l[np] < lMin) lMin = l[np];
            }
            if (c > 0) {
                // bottom left: g
                np = (r+1) * w + (c-1);
                if (f[r+1][c-1] == fMin) {
                   if (l[np] > 0 && l[np] < lMin) lMin = l[np];
                }
            }
            if (c < w - 1) {
                // bottom right: e
                np = (r+1) * w + (c+1);
                if (f[r+1][c+1] == fMin) {
                    if (l[np] > 0 && l[np] < lMin) lMin = l[np];
                }
            }
        }
        if (c > 0) {
            // left: h
            np = r * w + (c-1);
            if (f[r][c-1] == fMin) {
                if (l[np] > 0 && l[np] < lMin) lMin = l[np];
            }
        }
        if (c < w - 1) {
            // right: d
            np = r * w + (c+1);
            if (f[r][c+1] == fMin) {
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
            if (f[r-1][c] == f[r][c]) {
                if (v[np] == v[p]-1 && l[np] > 0 && l[np] < lMin) lMin = l[np];
            }
            if (c > 0) {
                // top left: a
                np = (r-1) * w + (c-1);
                if (f[r-1][c-1] == f[r][c]) {
                    if (v[np] == v[p]-1 && l[np] > 0 && l[np] < lMin) lMin = l[np];
                }
            }
            if (c < w - 1) {
                // top right: c
                np = (r-1) * w + (c+1);
                if (f[r-1][c+1] == f[r][c]) {
                    if (v[np] == v[p]-1 && l[np] > 0 && l[np] < lMin) lMin = l[np];
                }
            }
        }
        if (r < h - 1) {
            // below: f
            np = (r+1) * w + c; // neighbor p
            if (f[r+1][c] == f[r][c]) {
                if (v[np] == v[p]-1 && l[np] > 0 && l[np] < lMin) lMin = l[np];
            }
            if (c > 0) {
                // bottom left: g
                np = (r+1) * w + (c-1);
                if (f[r+1][c-1] == f[r][c]) {
                   if (v[np] == v[p]-1 && l[np] > 0 && l[np] < lMin) lMin = l[np];
                }
            }
            if (c < w - 1) {
                // bottom right: e
                np = (r+1) * w + (c+1);
                if (f[r+1][c+1] == f[r][c]) {
                    if (v[np] == v[p]-1 && l[np] > 0 && l[np] < lMin) lMin = l[np];
                }
            }
        }
        if (c > 0) {
            // left: h
            np = r * w + (c-1);
            if (f[r][c-1] == f[r][c]) {
                if (v[np] == v[p]-1 && l[np] > 0 && l[np] < lMin) lMin = l[np];
            }
        }
        if (c < w - 1) {
            // right: d
            np = r * w + (c+1);
            if (f[r][c+1] == f[r][c]) {
                if (v[np] == v[p]-1 && l[np] > 0 && l[np] < lMin) lMin = l[np];
            }
        }

    }

    // setting label for this pixel
    if (lMin != LMAX && l[p] != lMin) {
        l[p] = lMin;
        modified = 1;
    }
}

int main(int argc, char** argv) {
    struct timespec start, stop; 
	double time;

    // ALGORITHM IMPLEMENTATION BEGINS ----------------------------------------------

    // INITIALIZATION 
    // setting v[p] values matrix to 0s
    v = (unsigned char*) malloc (sizeof(unsigned char)*h*w);
    memset(v, 0, h*w*sizeof(char*));
    
    // setting l[p] labels matrix to 0s
    l = (unsigned char*) malloc (sizeof(unsigned char)*h*w);
    memset(l, 0, h*w*sizeof(char*));

    currentLabel = 1;
    // printing to test
    printf("f before step 1:\n");
    for (int row = 0; row < h; row++) {
        for (int col = 0; col < w; col++) {
            printf("%d ", f[row][col]);
        }
        printf("\n");
    }

    // STARTING TIME
    if( clock_gettime(CLOCK_REALTIME, &start) == -1) { perror("clock gettime");}

    // scan from top-left to bottom-right (p)
        // step1(p)
    for (int row = 0; row < h; row++) {
        for (int col = 0; col < w; col++) {
            step1(row, col);
        }
    }
    
    printf("v after step 1:\n");
    for (int row = 0; row < h; row++) {
        for (int col = 0; col < w; col++) {
            printf("%d ", v[row * w + col]);
        }
        printf("\n");
    }

    // executing step2
    while (1) {
        // scan from top-left to bottom-right(p)
            // step2(p)
        modified = 0;
        for (int row = 0; row < h; row++) {
            for (int col = 0; col < w; col++) {
                step2(row, col);
            }
        }

        // if v[p] is not modified
            // break
        if (!modified) {
            break;
        }

        // scan from bottom-right to top-left(p)
            // step2(p)
        modified = 0;
        for (int row = h-1; row >= 0; row--) {
            for (int col = w-1; col >= 0; col--) {
                step2(row, col);
            }
        }
        
        // if v[p] is not modified
            // break
        if (!modified) {
            break;
        }

    }

    printf("\n\nv after step 2:\n");
    for (int row = 0; row < h; row++) {
        for (int col = 0; col < w; col++) {
            printf("%d ", v[row * w + col]);
        }
        printf("\n");
    }


    // executing step3
    while (1) {
        // scan from top-left to bottom-right(p)
            // step3(p)
        modified = 0;
        for (int row = 0; row < h; row++) {
            for (int col = 0; col < w; col++) {
                step3(row, col);
            }
        }

        // if l[p] is not modified
            // break
        if (!modified) {
            break;
        }

        // scan from bottom-right to top-left(p)
            // step3(p)
        modified = 0;
        for (int row = h-1; row >= 0; row--) {
            for (int col = w-1; col >= 0; col--) {
                step3(row, col);
            }
        }

        // if l[p] is not modified
            // break
        if (!modified) {
            break;
        }
    }

    printf("\n\nl after step 3:\n");
    for (int row = 0; row < h; row++) {
        for (int col = 0; col < w; col++) {
            printf("%d ", l[row * w + col]);
        }
        printf("\n");
    }

    // STOPPING TIME
    if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror("clock gettime");}		
	time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e6;
    printf("Serial execution time = %f ms\n", time);	

    return 0;
}