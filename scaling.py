import os
import platform
"""
change the frequecy of cpu to speed up the execution.
"""
def scale():
    if platform.processor()=="aarch64":
        cmd1="echo 1400000 >/sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq & echo 1400000 >/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq"

        os.system(cmd1)
    print("Success!!")
