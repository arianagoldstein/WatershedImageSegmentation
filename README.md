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