import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# --- 1. НАСТРОЙКИ ---
FILE_PATH = 'exp1.csv'                  # Исходный файл
OUTPUT_DIR = 'data_files'               # Папка для сохранения результатов

# Создаем папку data_files, если она еще не существует
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUT_OSC_CSV = os.path.join(OUTPUT_DIR, 'oscillations.csv')
OUT_DERIV_CSV = os.path.join(OUTPUT_DIR, 'derivative.csv')

TIME_COL = 0
VOLT_COL = 1
START_TIME = 1.5

# --- 2. ЗАГРУЗКА ДАННЫХ ---
df = pd.read_csv(FILE_PATH)
t = df.iloc[:, TIME_COL].to_numpy(dtype=float)
U = df.iloc[:, VOLT_COL].to_numpy(dtype=float)

mask = t >= START_TIME
t = t[mask]
U = U[mask]

# --- 3. ОБРАБОТКА СИГНАЛА И ПОИСК МАКСИМУМОВ ---
dt = np.median(np.diff(t))
win_trend = int(1.5 / dt) | 1
win_smooth = max(5, int(0.15 / dt) | 1)

trend = savgol_filter(U, window_length=win_trend, polyorder=3)
oscillations = U - trend
osc_smooth = savgol_filter(oscillations, window_length=win_smooth, polyorder=2)

dU_dt = np.gradient(osc_smooth, t)
zero_crossings = np.where((dU_dt[:-1] > 0) & (dU_dt[1:] <= 0))

peak_times = t[zero_crossings]
peak_volts = U[zero_crossings]
peak_osc = osc_smooth[zero_crossings]
distances = np.diff(peak_times)

# --- 4. СОХРАНЕНИЕ В ПАПКУ data_files ---
pd.DataFrame({
    'time_s': t,
    'oscillations_V': osc_smooth
}).to_csv(OUT_OSC_CSV, index=False, sep=';', decimal=',')

pd.DataFrame({
    'time_s': t,
    'derivative_V_s': dU_dt
}).to_csv(OUT_DERIV_CSV, index=False, sep=';', decimal=',')

print(f"Файлы сохранены в папку '{OUTPUT_DIR}':\n1) {OUT_OSC_CSV}\n2) {OUT_DERIV_CSV}")
print(f"Найдено максимумов: {len(peak_times)}, средний период: {np.mean(distances):.4f} с")

# --- 5. ПОСТРОЕНИЕ ГРАФИКОВ ---
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

ax1.plot(t, U, label='Исходное напряжение', color='blue', alpha=0.7)
ax1.plot(t, trend, label='Выделенный тренд', color='orange', linestyle='--')
ax1.scatter(peak_times, peak_volts, s=1, color='red', zorder=5, label=f'Максимумы ({len(peak_times)} шт.)')
ax1.set_ylabel('Напряжение, В')
ax1.set_title('Поиск максимумов пульсаций через нули производной')
ax1.legend()
ax1.grid(True, linestyle=':', alpha=0.7)

ax2.plot(t, osc_smooth, label='Колебания без тренда', color='green')
ax2.scatter(peak_times, peak_osc, s=1, color='red', zorder=5)
ax2.axhline(0, color='black', linewidth=0.8)
ax2.set_ylabel('Амплитуда, В')
ax2.legend()
ax2.grid(True, linestyle=':', alpha=0.7)

ax3.plot(t, dU_dt, label='Производная dU/dt', color='purple')
ax3.axhline(0, color='black', linewidth=1, linestyle='--')
ax3.scatter(peak_times, np.zeros_like(peak_times), s=1, color='red', zorder=5, label='Пересечения с 0 (максимумы)')
ax3.set_xlabel('Время, с')
ax3.set_ylabel('dU/dt, В/с')
ax3.legend()
ax3.grid(True, linestyle=':', alpha=0.7)

plt.tight_layout()
plt.show()
