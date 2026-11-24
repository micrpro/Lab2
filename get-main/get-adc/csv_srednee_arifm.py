import csv
import os

# =====================================================================
# НАСТРОЙКИ: ВПИШИТЕ НАЗВАНИЯ ВАШИХ 9 ФАЙЛОВ НИЖЕ
# =====================================================================

INPUT_FILES = [
    'values40.csv',
    'values60.csv',
    'values80.csv',
    'values100.csv',
    'values120.csv',
    'values140.csv',
    'values160.csv',
    'values180.csv',
    'values200.csv',
]

OUTPUT_FOLDER = 'data_files'         # Папка для сохранения результата
OUTPUT_FILE = 'final_koeff.csv' # Имя файла внутри этой папки
MAX_TIME = 2.0                       # Учитываем измерения только до 2-й секунды включительно
DELIMITER = ','                      # Разделитель столбцов (',' или ';')

# =====================================================================
# ОСНОВНОЙ КОД
# =====================================================================

def get_avg_y_before_time(file_path, max_time, delimiter):
    """Считывает Y для всех строк, где X <= max_time, и возвращает среднее."""
    if not os.path.exists(file_path):
        print(f"[Ошибка] Файл не найден: {file_path}")
        return None

    y_coords = []

    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=delimiter)
        
        for row in reader:
            if len(row) < 2:
                continue
            try:
                # row[0] — время (X), row[1] — значение (Y)
                x = float(row[0].strip())
                y = float(row[1].strip())
                
                # Берем только измерения до 2-й секунды
                if x <= max_time:
                    y_coords.append(y)
            except ValueError:
                continue

    if not y_coords:
        return None

    return sum(y_coords) / len(y_coords)

def clean_filename(file_path):
    """Убирает путь, расширение .csv и слово 'values' из имени файла."""
    base_name = os.path.basename(file_path)            
    # Берем первый элемент [0] из кортежа (имя без расширения)
    name_without_ext = os.path.splitext(base_name)[0]  
    cleaned = name_without_ext.replace('values', '').strip('_- ')
    return cleaned

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    full_output_path = os.path.join(OUTPUT_FOLDER, OUTPUT_FILE)

    with open(full_output_path, mode='w', encoding='utf-8', newline='') as out_file:
        writer = csv.writer(out_file, delimiter=DELIMITER)
        writer.writerow(['Номер файла', 'Среднее Y'])

        for file_path in INPUT_FILES:
            file_label = clean_filename(file_path)
            avg_y = get_avg_y_before_time(file_path, MAX_TIME, DELIMITER)
            
            if avg_y is not None:
                writer.writerow([file_label, round(avg_y, 6)])
                print(f"[Готово] Файл '{file_label}' -> Среднее Y: {avg_y:.6f}")
            else:
                writer.writerow([file_label, 'Нет данных'])
                print(f"[Пропущен] Файл '{file_path}' (нет данных до {MAX_TIME} сек)")

    print(f"\nГотово! Результаты сохранены в: {full_output_path}")

if __name__ == '__main__':
    main()
