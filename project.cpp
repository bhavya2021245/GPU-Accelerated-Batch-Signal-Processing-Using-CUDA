#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>

constexpr int kNumSignals = 100;
constexpr int kSignalLength = 512;

// CUDA kernel: Normalizes each signal and finds the peak value
__global__ void ProcessSignalsKernel(const float* device_signals, float* device_peaks, int signal_length) {
  int signal_id = blockIdx.x;
  int thread_id = threadIdx.x;

  __shared__ float shared_signal[kSignalLength];
  __shared__ float shared_mean;
  __shared__ float shared_peak;

  int global_index = signal_id * signal_length + thread_id;

  if (thread_id < signal_length) {
    shared_signal[thread_id] = device_signals[global_index];
  }
  __syncthreads();

  // Calculate mean
  float local_sum = 0.0f;
  for (int i = thread_id; i < signal_length; i += blockDim.x) {
    local_sum += shared_signal[i];
  }

  if (thread_id == 0) shared_mean = 0.0f;
  __syncthreads();

  atomicAdd(&shared_mean, local_sum / signal_length);
  __syncthreads();

  // Normalize
  if (thread_id < signal_length) {
    shared_signal[thread_id] -= shared_mean;
  }
  __syncthreads();

  // Find peak
  if (thread_id == 0) shared_peak = 0.0f;
  __syncthreads();

  if (thread_id < signal_length) {
    float value = fabsf(shared_signal[thread_id]);
    atomicMax(reinterpret_cast<int*>(&shared_peak), __float_as_int(value));
  }
  __syncthreads();

  if (thread_id == 0) {
    device_peaks[signal_id] = shared_peak;
  }
}

int main() {
  int num_signals = kNumSignals;
  int signal_length = kSignalLength;
  int total_size = num_signals * signal_length;

  // Host allocations
  float* host_signals = reinterpret_cast<float*>(malloc(sizeof(float) * total_size));
  float* host_peaks = reinterpret_cast<float*>(malloc(sizeof(float) * num_signals));

  // Initialize signal data
  for (int i = 0; i < total_size; ++i) {
    host_signals[i] = static_cast<float>(rand() % 1000) / 100.0f;
  }

  // Device allocations
  float* device_signals = nullptr;
  float* device_peaks = nullptr;
  cudaMalloc(&device_signals, sizeof(float) * total_size);
  cudaMalloc(&device_peaks, sizeof(float) * num_signals);

  cudaMemcpy(device_signals, host_signals, sizeof(float) * total_size, cudaMemcpyHostToDevice);

  // Launch kernel
  ProcessSignalsKernel<<<num_signals, signal_length>>>(device_signals, device_peaks, signal_length);
  cudaDeviceSynchronize();

  cudaMemcpy(host_peaks, device_peaks, sizeof(float) * num_signals, cudaMemcpyDeviceToHost);

  // Print results
  for (int i = 0; i < 5; ++i) {
    printf("Signal %d peak after normalization: %.4f\n", i, host_peaks[i]);
  }

  // Cleanup
  cudaFree(device_signals);
  cudaFree(device_peaks);
  free(host_signals);
  free(host_peaks);
  cudaDeviceReset();

  return 0;
}
