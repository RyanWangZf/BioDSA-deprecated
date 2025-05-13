import math

"""
Utility function to preview the runtime of experiments.
"""

log_dir = "experiment_logs"
timing_file = f"~/DSWizard/{log_dir}/timing_log.txt"

# read the file line by line. 
times = []
with open(timing_file, 'r') as file:
    for line in file:
        line = line.strip()
        if line == "0.00":
            continue
        times.append(float(line))

# sort the times
times = sorted(times)

# print the median
print("Median: ", times[len(times) // 2])

# print the mean
print("Mean: ", sum(times) / len(times))

# print the standard deviation
print("Standard Deviation: ", math.sqrt(sum((x - sum(times) / len(times)) ** 2 for x in times) / len(times)))
