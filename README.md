# Parallel Implementation of Watershed Image Segmentation Algorithms
Ariana Goldstein and Iris Li


We developed the following two implementations of the watershed image segmentation algorithm.

## Scanning Algorithm
We implemented this algorithm serially but haven't fully parallelized it yet. Here are the steps to to compile and run the serial implementation.

**With basic input (small 2D matrix):** In the `serial_scanning_basicinput.c` file, change the declaration of `f` on line 20 to a different input array and define `h` and `w` (height and width) on lines 13 and 14 accordingly. Then, in the `scanning_algorithm` directory, run the following terminal commands.
```
$ make ssb
$ ./ssb
```
To visualize the resulting segmentation in color, copy this input into a text file `input.txt`, and you can run `colorize.py`.
```
$ python colorize.py
```

**With actual image input:** In the `serial_scanning.c`file, change the input file declaration on line 16 to the path of your input image. Define `h` and `w` accordingly. Then, run the following terminal commands.
```
$ make ss
$ ./ss
```


## Altitude Algorithm
To run the Altitude algorithm without shared memory, in CARC type:
```
$ module load gcc/8.3.0
$ module load nvidia-hpc-sdk
$ make cuda2
$ sbatch job1.sl
```

To run the algorithm with shared memory:
```
$ module load gcc/8.3.0
$ module load nvidia-hpc-sdk
$ make cuda3
$ sbatch job.sl
```

To see the run time for either, enter:
```
$ cat gpujob.out
```

To see the output, download `output.raw`

To change input files, open cuda3.cu or cuda2.cu and change input_file on line 9. 
Also change the height and width on lines 6 and 7. 
The following are the input names and their sizes:
|Name		|height		|width	|
|:----------|:----------|:------|
|input.raw	|800		|800	|
|512.raw	|512		|512	|
|woman.raw	|256		|256	|