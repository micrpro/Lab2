import matplotlib.pyplot as plt
import pandas as pd

# Добавляем decimal=',' чтобы запятая в числах воспринималась как дробная часть
df = pd.read_csv('data_files/oscillations.csv', sep=';', skiprows=1, header=None, decimal=',')

# Берем первый (0) и второй (1) столбцы
x = df[0]
y = df[1]

plt.figure(figsize=(10, 6))
plt.plot(x, y, marker='x', markersize=1, linestyle='-', color='b')

plt.title('График по данным из CSV')
plt.xlabel('Ось X')
plt.ylabel('Ось Y')
plt.grid(True)

plt.show()