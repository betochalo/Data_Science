import numpy as np
import requests
import json
import matplotlib.pyplot as plt
from pprint import pprint

url = "https://api.open-meteo.com/v1/forecast?latitude=0.4183&longitude=-78.1931&hourly=temperature_2m&timezone=auto"

x = requests.get(url)
data = x.json()
hourly = data.get('hourly', {})
time = hourly.get('time', [])
temperature = hourly.get('temperature_2m', [])
print(time)
plt.plot(np.arange(len(time)), temperature)
plt.title("Temperature of Urcuqui")
plt.xlabel("Time")
plt.ylabel("Temperature °C")
plt.grid(axis="y")
# plt.savefig("Images/Temperature_Urcuqui")
