from pynvml import *

nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)  # Index of your GPU
# pid = os.getpid()  # Current process ID
processes = nvmlDeviceGetComputeRunningProcesses(handle)

print(len(processes))
for proc in processes:
    # if proc.pid == pid:
    print(proc.usedGpuMemory)
    print(f"Memory Used: {proc.usedGpuMemory / 1024**2:.2f} MB")
nvmlShutdown()
