# GPU-Accelerated-Batch-Signal-Processing-Using-CUDA

Code Project Description:
This project implements a CUDA-based solution to perform batch signal processing on a large set of CSV-based signal data. The aim was to efficiently analyze hundreds of signal samples (time-series data) using parallel processing on the GPU. Each signal sample consists of values like voltages or sensor readings over time.

The CUDA kernel performs a combination of basic digital signal processing (DSP) tasks on each sample:

Noise reduction (simple moving average filter)

Normalization

Peak detection

The host code loads multiple CSV files, copies the data to the GPU, and launches kernels to process all signal sets concurrently. This parallel strategy leverages the GPU’s massive thread capacity, allowing us to process hundreds of small signals in parallel with minimal latency.

Thought Process and Development:
The goal was to design something scalable — a system that could easily handle either many small inputs or fewer large inputs by tuning block and grid configurations. After exploring several problem domains, signal processing stood out as something where:

Each input is independent (embarrassingly parallel)

There's clear value in acceleration (e.g., for sensor analytics, IoT, or biomedical signals)

I structured the kernel to operate on one signal per thread block, with each thread processing a slice of that signal. Shared memory was used for intermediate values to reduce global memory latency.

Issues Encountered:
Initially, kernel memory access patterns caused race conditions during reduction/peak detection. These were fixed by proper use of thread synchronization (__syncthreads()) and restructuring calculations to avoid overlapping writes.

I also ran into memory copying errors when trying to process large batches. This was resolved by batching the data dynamically and tuning the device memory allocation.

Lessons Learned:
Thread indexing strategies are critical in high-throughput signal/image applications.

Proper memory layout and access patterns (e.g., coalesced access) greatly improve performance.

CUDA streams (considered for future work) could allow even better concurrency for overlapping computation and data transfer.


