import pynvml

def print_gpu_memory_usage(gpu_index=0):
    """
    打印指定 GPU 的显存使用情况。

    参数:
        gpu_index (int): GPU 索引，默认值为 0。
    """
    try:
        # 初始化 NVML
        pynvml.nvmlInit()

        # 获取 GPU 的句柄
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

        # 获取 GPU 的名称
        gpu_name = pynvml.nvmlDeviceGetName(handle)

        # 获取 GPU 的显存信息
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

        # 打印显存占用情况在一行
        print(f"GPU {gpu_index} ({gpu_name.decode('utf-8')}): Total={mem_info.total / 1024 ** 2:.2f} MB,"
              f" Used={mem_info.used / 1024 ** 2:.2f} MB, Free={mem_info.free / 1024 ** 2:.2f} MB")

    except pynvml.NVMLError as error:
        print(f"Failed to retrieve GPU data: {error}")

    finally:
        # 清理 NVML
        pynvml.nvmlShutdown()

if __name__ == '__main__':
    print_gpu_memory_usage()
