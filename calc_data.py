import sys
import os
import mmap
import numpy as np
from concurrent.futures import ThreadPoolExecutor


def process_chunk(args):
    filepath, byte_start, byte_end = args
    
    byte_size = byte_end - byte_start
    if byte_size == 0:
        return None
    
    with open(filepath, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        
        arr = np.frombuffer(
            mm,
            dtype=">u4",
            count=byte_size // 4,
            offset=byte_start
        )
        
        local_sum = 0
        block_size = 4000000
        
        for i in range(0, len(arr), block_size):
            local_sum += int(arr[i:i + block_size].sum(dtype=np.uint64))
        
        local_min = int(arr.min())
        local_max = int(arr.max())
        
        del arr
        mm.close()
        
        return local_sum, local_min, local_max


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 calc_data.py <path_to_file>", file=sys.stderr)
        sys.exit(1)

    filepath = sys.argv[1]

    if not os.path.isfile(filepath):
        print(f"File not found: {filepath}", file=sys.stderr)
        sys.exit(1)

    file_size = os.path.getsize(filepath)

    if file_size == 0:
        print("sum=0\nmin=0\nmax=0")
        return

    num_cpus = os.cpu_count() or 1
    chunk_size = 256 * 1024 * 1024

    chunks = []

    for byte_start in range(0, file_size, chunk_size):
        byte_end = min(byte_start + chunk_size, file_size)
        byte_end -= byte_end % 4
        byte_size = byte_end - byte_start

        if byte_size > 0:
            chunks.append((filepath, byte_start, byte_end))

    total_sum = 0
    total_min = None
    total_max = None

    with ThreadPoolExecutor(max_workers=num_cpus) as executor:
        results = list(executor.map(process_chunk, chunks))

    for res in results:
        if res is None:
            continue
        s, mn, mx = res
        total_sum += s
        if total_min is None or mn < total_min:
            total_min = mn
        if total_max is None or mx > total_max:
            total_max = mx

    if total_min is None:
        total_min = 0
        total_max = 0

    print(f"sum={total_sum}")
    print(f"min={total_min}")
    print(f"max={total_max}")


if __name__ == "__main__":
    main()
