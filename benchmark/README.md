# Benchmark

On CPU, using Ubuntu 16.04, using float32, Torch LSTM boasts 900 samples/sec compared to TF’s 809 samples/sec for LSTM with 512 hiddensize and 64 batchsize.
On the other hand, for 128 hiddensize and 32 batchsize, Torch has 3990 compared to TF’s 4130 samples/sec.