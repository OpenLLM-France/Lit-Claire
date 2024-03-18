import subprocess
import socket
import os

def run_command(command, need_gpu=False, doit=True):
    if isinstance(command, list):
        command = " ".join(command)
    
    if need_gpu:
        jeanzay = socket.gethostname().startswith("jean-zay")
        if jeanzay:
            prefix = "srun --ntasks=1 --gres=gpu:1 --constraint=a100 --account=qgz@a100 --cpus-per-task=8 --qos=qos_gpu-dev "
        else:
            prefix = ""
        command = prefix + command

    print(f"... Running command ...\n{command}")
    if not doit:
        return
    
    return subprocess.run(command, shell=True, check=True)
