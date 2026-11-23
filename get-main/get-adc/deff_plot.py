import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# --- 1. НАСТРОЙКИ (измените под свой файл) ---
FILE_PATH = 'exp1.csv'     # Имя вашего CSV-файла
TIME_COL = 0               # Индекс (или название) столбца со временем
VOLT_COL = 1               # Индекс (или название) столбца с напряжением
START_TIME = 1.5           # Отсекаем начальный плоский участок до 1.5 с (если нужно)

# --- 2. ЗАГРУЗКА ДАННЫХ ---
# Если разделитель точка с запятой, добавьте параметр sep=';'
df = pd.read_csv(FILE_PATH)

# Берем столбцы по индексу (если указали имена столбцов, замените iloc на df[TIME_COL])
t = df.iloc[:, TIME_COL].to_numpy(dtype=float)
U = df.iloc[:, VOLT_COL].to_numpy(dtype=float)

# Оставляем только участок после начала спада
mask = t >= START_TIME
t = t[mask]
U = U[mask]

# --- 3. ПОИСК МАКСИМУМОВ ЧЕРЕЗ ПРОИЗВОДНУЮ ---
# Автоматически подбираем окна сглаживания под количество точек
dt = np.median(np.diff(t))
win_trend = int(1.5 / dt) | 1    # Окно ~1.5 секунды (нечетное) для выделения тренда
win_smooth = max(5, int(0.15 / dt) | 1)  # Окно ~0.15 секунды для очистки от шума

# Выделяем тренд и получаем чистые колебания вокруг нуля
trend = savgol_filter(U, window_length=win_trend, polyorder=3)
oscillations = U - trend
osc_smooth = savgol_filter(oscillations, window_length=win_smooth, polyorder=2)

# Считаем производную и ищем пересечения с нулем сверху вниз (с "+" на "-")
dU_dt = np.gradient(osc_smooth, t)
zero_crossings = np.where((dU_dt[:-1] > 0) & (dU_dt[1:] <= 0))[0]

peak_times = t[zero_crossings]
peak_volts = U[zero_crossings]
distances = np.diff(peak_times)

# --- 4. ВЫВОД РЕЗУЛЬТАТОВ В КОНСОЛЬ ---
print(f"Найдено максимумов: {len(peak_times)}")
print(f"Среднее расстояние между максимумами (период): {np.mean(distances):.4f} с")
print(f"Стандартное отклонение периода: {np.std(distances):.4f} с")

# --- 5. ПОСТРОЕНИЕ ГРАФИКА ---
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# График 1: Исходный сигнал и найденные максимумы
ax1.plot(t, U, label='Исходное напряжение', color='blue', alpha=0.7)
ax1.plot(t, trend, label='Выделенный тренд', color='orange', linestyle='--')
ax1.scatter(peak_times, peak_volts, s=1, color='red', zorder=5, label=f'Максимумы ({len(peak_times)} шт.)')
ax1.set_ylabel('Напряжение, В')
ax1.set_title('Поиск максимумов пульсаций через нули производной')
ax1.legend()
ax1.grid(True, linestyle=':', alpha=0.7)

# График 2: Колебания без тренда и их производная
ax2.plot(t, osc_smooth, label='Колебания без тренда', color='green')
ax2.scatter(peak_times, osc_smooth[zero_crossings], s=1,  color='red', zorder=5)
ax2.axhline(0, color='black', linewidth=0.8)
ax2.set_ylabel('Амплитуда, В')
ax2.legend()
ax2.grid(True, linestyle=':', alpha=0.7)

# График 3: Производная и точки пересечения с нулем
ax3.plot(t, dU_dt, label='Производная dU/dt', color='purple')
ax3.axhline(0, color='black', linewidth=1, linestyle='--')
ax3.scatter(peak_times, np.zeros_like(peak_times), s=1, color='red', zorder=5, label='Пересечения с 0 (максимумы)')
ax3.set_xlabel('Время, с')
ax3.set_ylabel('dU/dt, В/с')
ax3.legend()
ax3.grid(True, linestyle=':', alpha=0.7)

plt.tight_layout()
plt.show()
