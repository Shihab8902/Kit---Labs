"""Check env + GPU benchmark. All paths stay on D:."""
import shutil
import sys
import time
import platform
import torch


def gb(n):
    return round(n / (1024 ** 3), 1)


def main():
    print(f"python={sys.version.split()[0]} exe={sys.executable}")
    print(f"torch={torch.__version__} cuda_build={torch.version.cuda}")
    print(f"cuda_ok={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}")
        free, total = torch.cuda.mem_get_info(0)
        print(f"vram_total_gb={gb(total)} vram_free_gb={gb(free)}")
    import os
    print(f"cpu={platform.processor()} count={os.cpu_count()}")
    try:
        import psutil
    except ImportError:
        psutil = None
    if psutil:
        print(f"ram_gb={gb(psutil.virtual_memory().total)}")
    else:
        print("ram=unknown (psutil not installed)")
    for d in ["C:\\", "D:\\"]:
        try:
            total, used, free = shutil.disk_usage(d)
            print(f"disk_{d[0]}_free_gb={gb(free)}")
        except Exception as e:
            print(f"disk_{d[0]}=error {e}")

    n = 4096
    print(f"benchmark matmul {n}x{n} x10 iters")
    a = torch.randn(n, n)
    b = torch.randn(n, n)
    t0 = time.time()
    for _ in range(10):
        _ = a @ b
    print(f"cpu_time_s={round(time.time() - t0, 2)}")

    if torch.cuda.is_available():
        a = a.cuda()
        b = b.cuda()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10):
            _ = a @ b
        torch.cuda.synchronize()
        print(f"gpu_time_s={round(time.time() - t0, 2)}")


if __name__ == "__main__":
    main()
