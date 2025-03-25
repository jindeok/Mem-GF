import psutil
import subprocess
import threading
import time
import os
from datetime import datetime

# Global flag to control logging
logging_active = True

def log_ram_peak(pid, model_name, output_dir="mem_logging", interval=1):
    global logging_active
    peak_ram = 0  # Peak RAM in MB

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Construct output file name
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"{model_name}_{date_str}_ram.txt")

    with open(output_file, "w") as log_file:
        log_file.write("Monitoring RAM usage for PID {}...\n".format(pid))
        try:
            process = psutil.Process(pid)
            while logging_active:
                # Get current RAM usage for the specific process in MB
                current_ram = process.memory_info().rss / (1024 * 1024)
                peak_ram = max(peak_ram, current_ram)

                # Log current and peak RAM usage
                log_file.write(f"Current RAM: {current_ram:.2f} MB, Peak RAM: {peak_ram:.2f} MB\n")
                log_file.flush()
                time.sleep(interval)
        except psutil.NoSuchProcess:
            log_file.write("Process ended.\n")

        log_file.write(f"Final Peak RAM: {peak_ram:.2f} MB\n")
        print(f"Final Peak RAM: {peak_ram:.2f} MB")

def log_vram_peak(model_name, cuda_device=1, output_dir="mem_logging", interval=1):
    global logging_active
    peak_vram = 0  # Peak VRAM in MB

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Construct output file name
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"{model_name}_{date_str}_vram.txt")

    with open(output_file, "w") as log_file:
        log_file.write(f"Monitoring VRAM usage for CUDA:{cuda_device}...\n")
        while logging_active:
            # Use nvidia-smi to get GPU memory usage
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if result.returncode != 0:
                log_file.write(f"Error querying GPU: {result.stderr}\n")
                break

            # Parse VRAM usage in MB
            gpu_usage = result.stdout.strip().split("\n")
            current_vram = int(gpu_usage[cuda_device])
            peak_vram = max(peak_vram, current_vram)

            # Log current and peak VRAM usage
            log_file.write(f"Current VRAM: {current_vram} MB, Peak VRAM: {peak_vram:.2f} MB\n")
            log_file.flush()
            time.sleep(interval)

        log_file.write(f"Final Peak VRAM: {peak_vram:.2f} MB\n")
        print(f"Final Peak VRAM: {peak_vram:.2f} MB")

def main():
    global logging_active

    # Arguments for main.py
    arguments = ["--dataset", "twitch", "--model", "turbo-cf", "--device", "cpu", "--filter", "1"]

    # Extract model name from arguments
    model_name = arguments[arguments.index("--model") + 1]

    # Run the main.py script with arguments and get its PID
    process = subprocess.Popen(["python", "main.py"] + arguments)
    pid = process.pid

    # Threads for RAM and VRAM logging
    ram_thread = threading.Thread(target=log_ram_peak, args=(pid, model_name))
    vram_thread = threading.Thread(target=log_vram_peak, args=(model_name, 1))

    # Start logging threads
    ram_thread.start()
    vram_thread.start()

    # Wait for the main.py process to complete
    process.wait()

    # Stop logging after main.py completes
    logging_active = False

    # Ensure threads terminate
    ram_thread.join()
    vram_thread.join()

if __name__ == "__main__":
    main()
