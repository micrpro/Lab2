import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

# =====================================================================
# === НАСТРОЙКИ ПОЛЬЗОВАТЕЛЯ (МЕНЯЙТЕ ТОЛЬКО ЭТОТ БЛОК) ===============
# =====================================================================

# 1. Пути к папке и файлам
FOLDER_PATH = "data_files"
INPUT_FILENAME = "derivative.csv"
OUTPUT_CSV_FILENAME = "new_derivative.csv"
OUTPUT_PLOT_FILENAME = "pulse_plot.png"

# 2. Параметры чтения и записи CSV
SKIP_ROWS = 1
INPUT_CSV_SEP = ";"
DECIMAL_SEPARATOR = ","
OUTPUT_CSV_SEP = ";"

# 3. Параметры сглаживания и поиска ГЛАВНЫХ пиков (оставили как на вашем графике)
SMOOTH_WINDOW = 35
MIN_HEART_PERIOD_SEC = 0.45
PRIMARY_PEAK_PROMINENCE = 0.15

# 4. Точные границы под ваши черные точки (включая желтую зону)
SECONDARY_MAX_VOLTAGE = 0.10  # Верхняя граница (захватывает верхние черные точки)
SECONDARY_MIN_VOLTAGE = (
    -0.06
)  # Нижняя граница (захватывает пики в желтой зоне)
MIN_DELAY_AFTER_PRIMARY_SEC = (
    0.11  # Начало окна поиска после красного пика (сек)
)
MAX_DELAY_AFTER_PRIMARY_SEC = 0.26  # Конец окна поиска после красного пика (сек)
SECONDARY_PEAK_PROMINENCE = 0.0002  # Высокая чувствительность к горбикам

# 5. Физические константы из методички
PRESSURE_COEFF = 1.0 / 0.0123
PATIENT_HEIGHT_M = 1.75

# 6. Настройка графика
SHOW_PLOT_WINDOW = True
PLOT_DPI = 300
MARKER_SIZE = 30

# =====================================================================
# === ОСНОВНОЙ КОД (ДАЛЬШЕ МОЖНО НЕ МЕНЯТЬ) ===========================
# =====================================================================


def analyze_pulse_wave():
    input_filepath = os.path.join(FOLDER_PATH, INPUT_FILENAME)
    output_csv_filepath = os.path.join(FOLDER_PATH, OUTPUT_CSV_FILENAME)
    output_plot_filepath = os.path.join(FOLDER_PATH, OUTPUT_PLOT_FILENAME)

    if not os.path.exists(input_filepath):
        raise FileNotFoundError(f"Файл '{input_filepath}' не найден!")

    df = pd.read_csv(
        input_filepath,
        skiprows=SKIP_ROWS,
        sep=INPUT_CSV_SEP,
        decimal=DECIMAL_SEPARATOR,
        header=None,
    )
    df = df.iloc[:, :2].apply(pd.to_numeric, errors="coerce").dropna()
    t = df.iloc[:, 0].to_numpy(dtype=float)
    u_raw = df.iloc[:, 1].to_numpy(dtype=float)

    dt_sample = np.median(np.diff(t))
    min_dist_samples = max(1, int(MIN_HEART_PERIOD_SEC / dt_sample))

    win = SMOOTH_WINDOW if SMOOTH_WINDOW % 2 == 1 else SMOOTH_WINDOW + 1
    win = min(win, len(u_raw) - (1 if len(u_raw) % 2 == 0 else 0))
    u_smooth = savgol_filter(u_raw, window_length=win, polyorder=4)

    primary_peaks, _ = find_peaks(
        u_smooth, prominence=PRIMARY_PEAK_PROMINENCE, distance=min_dist_samples
    )
    if len(primary_peaks) < 2:
        raise ValueError("Найдено меньше 2 главных пиков!")

    secondary_peaks, min_troughs, dt_list, reflection_ratios = [], [], [], []

    for i in range(len(primary_peaks) - 1):
        p1, p2 = primary_peaks[i], primary_peaks[i + 1]
        local_min_idx = p1 + np.argmin(u_smooth[p1:p2])
        min_troughs.append(local_min_idx)

        # Выделяем точный временной интервал [0.11 .. 0.26 с] после главного пика
        idx_start = p1 + int(MIN_DELAY_AFTER_PRIMARY_SEC / dt_sample)
        idx_end = min(
            local_min_idx, p1 + int(MAX_DELAY_AFTER_PRIMARY_SEC / dt_sample)
        )

        if idx_end > idx_start + 2:
            window_u = u_smooth[idx_start:idx_end]
            # 1. Ищем явный локальный максимум внутри этого окна
            sec_candidates, _ = find_peaks(
                window_u, prominence=SECONDARY_PEAK_PROMINENCE
            )
            valid = [
                idx_start + idx
                for idx in sec_candidates
                if SECONDARY_MIN_VOLTAGE
                <= u_smooth[idx_start + idx]
                <= SECONDARY_MAX_VOLTAGE
            ]

            best_sec = None
            if valid:
                # Берем самый высокий пик в окне
                best_sec = valid[np.argmax(u_smooth[valid])]
            else:
                # 2. Если пик сглажен в "ступеньку" (как на крутом спаде),
                # берем точку максимальной выпуклости вниз (минимум второй производной)
                d2u = np.gradient(np.gradient(window_u))
                best_local = np.argmin(d2u)
                candidate_idx = idx_start + best_local
                if (
                    SECONDARY_MIN_VOLTAGE
                    <= u_smooth[candidate_idx]
                    <= SECONDARY_MAX_VOLTAGE
                ):
                    best_sec = candidate_idx

            if best_sec is not None:
                secondary_peaks.append(best_sec)
                dt_list.append(t[best_sec] - t[p1])

                base_u = u_smooth[local_min_idx]
                amp_p = u_smooth[p1] - base_u
                amp_s = u_smooth[best_sec] - base_u
                if amp_p > 0:
                    reflection_ratios.append(amp_s / amp_p)

    secondary_peaks = np.array(secondary_peaks, dtype=int)
    min_troughs = np.array(min_troughs, dtype=int)

    # Расчет показателей
    mean_rr = np.mean(np.diff(t[primary_peaks]))
    pulse_bpm = 60.0 / mean_rr
    mean_dt = np.mean(dt_list) if dt_list else float("nan")
    mean_ratio = (
        np.mean(reflection_ratios) if reflection_ratios else float("nan")
    )
    ri_pct = mean_ratio * 100.0
    si_val = (
        (PATIENT_HEIGHT_M / mean_dt)
        if (mean_dt and mean_dt > 0)
        else float("nan")
    )

    u_sys = np.mean(u_smooth[primary_peaks])
    u_dia = np.mean(u_smooth[min_troughs])
    p_sys, p_dia = u_sys * PRESSURE_COEFF, u_dia * PRESSURE_COEFF
    p_ratio = p_sys / p_dia if p_dia != 0 else float("nan")

    print("\n" + "=" * 55)
    print(f"Главных пиков: {len(primary_peaks)} | Вторичных: {len(secondary_peaks)}")
    print(f"Пульс: {pulse_bpm:.1f} уд/мин (T_sys = {mean_rr:.4f} с)")
    print(f"Задержка вторичного пика (dt): {mean_dt:.4f} с")
    print(f"Отношение пиков (U_мал/U_бол): {mean_ratio:.4f}")
    print(f"Коэф. отражения (RI): {ri_pct:.2f} %")
    print(f"Коэф. жесткости сосудов (SI): {si_val:.2f} м/с")
    print(
        f"АД: {p_sys:.1f} / {p_dia:.1f} мм рт. ст. (Отношение P_sys/P_dia: {p_ratio:.3f})"
    )
    print("=" * 55 + "\n")

    # Сохранение в CSV
    os.makedirs(FOLDER_PATH, exist_ok=True)
    rows = (
        [
            {
                "Time": t[i],
                "Voltage": u_smooth[i],
                "Pressure": u_smooth[i] * PRESSURE_COEFF,
                "Type": "Primary_Peak",
            }
            for i in primary_peaks
        ]
        + [
            {
                "Time": t[i],
                "Voltage": u_smooth[i],
                "Pressure": u_smooth[i] * PRESSURE_COEFF,
                "Type": "Secondary_Peak",
            }
            for i in secondary_peaks
        ]
        + [
            {
                "Time": t[i],
                "Voltage": u_smooth[i],
                "Pressure": u_smooth[i] * PRESSURE_COEFF,
                "Type": "Diastolic_Min",
            }
            for i in min_troughs
        ]
    )

    pd.DataFrame(rows).sort_values("Time").to_csv(
        output_csv_filepath,
        index=False,
        sep=OUTPUT_CSV_SEP,
        decimal=DECIMAL_SEPARATOR,
    )
    print(f"[OK] CSV сохранен: {output_csv_filepath}")

    # График (минимальные изменения)
    plt.figure(figsize=(13, 6))
    plt.plot(t, u_raw, label="Исходный АЦП", color="lightgray", linewidth=1.0)
    plt.plot(
        t, u_smooth, label="Сглаженный сигнал", color="royalblue", linewidth=1.5
    )

    plt.axhspan(
        SECONDARY_MIN_VOLTAGE,
        SECONDARY_MAX_VOLTAGE,
        color="orange",
        alpha=0.12,
        label=f"Зона поиска ({SECONDARY_MIN_VOLTAGE}..{SECONDARY_MAX_VOLTAGE} В)",
    )
    plt.axhline(
        SECONDARY_MAX_VOLTAGE, color="orange", linestyle=":", linewidth=1.0
    )
    plt.axhline(
        SECONDARY_MIN_VOLTAGE, color="orange", linestyle="--", linewidth=1.0
    )

    plt.scatter(
        t[primary_peaks],
        u_smooth[primary_peaks],
        s=MARKER_SIZE,
        color="red",
        zorder=5,
        label="Главные пики",
    )
    if len(secondary_peaks) > 0:
        plt.scatter(
            t[secondary_peaks],
            u_smooth[secondary_peaks],
            s=MARKER_SIZE,
            color="orange",
            zorder=5,
            label="Вторичные пики",
        )
    if len(min_troughs) > 0:
        plt.scatter(
            t[min_troughs],
            u_smooth[min_troughs],
            s=MARKER_SIZE,
            color="green",
            zorder=5,
            label="Минимумы",
        )

    plt.title(
        f"Пульс: {pulse_bpm:.1f} уд/мин | АД: {p_sys:.0f}/{p_dia:.0f} | инд. отражения: {ri_pct:.1f}% | Скорость пульсовой волны: {si_val:.2f} м/с"
    )
    plt.xlabel("Время, с")
    plt.ylabel("Напряжение, В")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_plot_filepath, dpi=PLOT_DPI)
    print(f"[OK] График сохранен: {output_plot_filepath}")

    if SHOW_PLOT_WINDOW:
        plt.show()


if __name__ == "__main__":
    analyze_pulse_wave()
