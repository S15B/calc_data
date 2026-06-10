import sys
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor


def process_chunk(args):
    filepath, byte_start, byte_end = args
    
    length = byte_end - byte_start
    if length == 0:
        return None
    
    with open(filepath, "rb") as f:
        f.seek(byte_start)
        data_bytes = f.read(length)
    
    arr = np.frombuffer(data_bytes, dtype='>u4')
    
    local_min = int(arr.min())
    local_max = int(arr.max())
    local_sum = int(np.sum(arr, dtype=np.uint64))
    
    return local_sum, local_min, local_max


def main():
    if len(sys.argv) < 2:
        print("Use: python3 calc_data.py <path_to_file>", file=sys.stderr)
        sys.exit(1)

    filepath = sys.argv[1]

    if not os.path.isfile(filepath):
        print(f"File not found: {filepath}", file=sys.stderr)
        sys.exit(1)

    file_size = os.path.getsize(filepath)

    if file_size == 0:
        print("sum=0\nmin=0\nmax=0")
        return

    num_elements = file_size // 4
    num_cpus = os.cpu_count() or 1
    num_workers = min(num_cpus * 2, max(1, (file_size // (100 * 1024 * 1024)) + 1))

    base = num_elements // num_workers
    remainder = num_elements % num_workers

    chunks = []
    start_elem = 0

    for i in range(num_workers):
        count = base + (1 if i < remainder else 0)
        if count == 0:
            continue
        byte_start = start_elem * 4
        byte_end = byte_start + count * 4
        chunks.append((filepath, byte_start, byte_end))
        start_elem += count

    with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
        results = list(executor.map(process_chunk, chunks))

    total_sum = 0
    total_min = None
    total_max = None

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