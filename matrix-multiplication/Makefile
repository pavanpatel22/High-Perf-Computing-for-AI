CC=gcc
CFLAGS=-O2 -pthread

all: test benchmark

test:
	$(CC) $(CFLAGS) src/single_thread.c src/multi_thread.c tests/test_correctness.c -o test

benchmark:
	$(CC) $(CFLAGS) src/multi_thread.c benchmark/benchmark.c -o benchmark

clean:
	rm -f test benchmark
