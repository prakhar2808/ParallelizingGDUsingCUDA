import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd
import numpy as np

data = pd.read_csv('sgd_parallel.csv')

timestamp = []
util = []

timestamp = data['timestamp'] 
util = pd.to_numeric(data[' utilization.gpu [%]'])

plt.figure()
#plt.scatter(range(len(timestamp)), util)
plt.plot(util)
plt.xlabel('Time Stamp')
plt.ylabel('GPU Utilization')
plt.title("GPU Utilization Curve")
plt.show()


'''
NVIDIA COMMAND

timeout 60 nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 1 > sgd_parallel.csv

'''