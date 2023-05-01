CC = gcc

ss: serial_scanning.c
	$(CC) -g serial_scanning.c -o ss

ssb: serial_scanning_basicinput.c
	$(CC) -g serial_scanning_basicinput.c -o ssb

ps: parallel_scanning.cu
	nvcc parallel_scanning.cu -o ps

clean:
	rm ss ssb ps