import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. Чтение данных из CSV-файла
# Предполагается, что в файле 2 столбца (X и Y) и есть заголовок в первой строке
filename = 'data_files/final_koeff.csv'
df = pd.read_csv(filename, sep=',')

# Извлекаем значения в numpy-массивы
x = df.iloc[:, 0].values  # Первый столбец (целые числа)
y = df.iloc[:, 1].values  # Второй столбец (числа с плавающей точкой)

# 2. Линейная аппроксимация (y = k*x + b)
# np.polyfit возвращает коэффициенты полинома 1-й степени: k (угловой) и b (свободный)
k, b = np.polyfit(x, y, 1)

# Вычисляем значения Y для линии аппроксимации
y_approx = k * x + b

# Выводим угловой коэффициент в консоль
print(f'Угловой коэффициент (k): {k:.4f}')
print(f'Уравнение прямой: y = {k:.4f}x + {b:.4f}')

# 3. Построение графика
plt.figure(figsize=(10, 6))

# Исходные точки
plt.scatter(x, y, color='blue', label='Исходные точки', zorder=5)

# Прямая аппроксимации
plt.plot(
    x,
    y_approx,
    color='red',
    linewidth=2,
    label=f'Аппроксимация: y = {k:.2f}x + {b:.2f}\nУгловой коэффициент (k) = {k:.4f}',
)

# Оформление графика
plt.title('Линейная аппроксимация данных из CSV')
plt.xlabel(df.columns[0])  # Название оси X берется из заголовка CSV
plt.ylabel(df.columns[1])  # Название оси Y берется из заголовка CSV
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Отображение графика
plt.tight_layout()
plt.show()
