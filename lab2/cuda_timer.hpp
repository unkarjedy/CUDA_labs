#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class CudaTimer {
public:
    CudaTimer();

    void start();
    void stop();

    float getTime();

private:
    cudaEvent_t m_start, m_stop;

};