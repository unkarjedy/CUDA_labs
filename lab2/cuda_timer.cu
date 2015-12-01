#include "cuda_timer.hpp"

CudaTimer::CudaTimer(){
    cudaEventCreate(&m_start);
    cudaEventCreate(&m_stop);
}


void CudaTimer::start(){
    cudaDeviceSynchronize();
    cudaEventRecord(m_start, 0);
}

void CudaTimer::stop(){
    cudaEventRecord(m_stop, 0);
    cudaEventSynchronize(m_stop);
    //cudaDeviceSynchronize();
}


float CudaTimer::getTime(){
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, m_start, m_stop);
    return elapsedTime;
}