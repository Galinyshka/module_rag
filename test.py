import os
import re

def rpd_list_generator(filename, lowercase=True, filter_empty=True):
    """
    Читает файл из домашней директории и нормализует строки.
    
    Args:
        filename: имя файла
        lowercase: приводить к нижнему регистру
        filter_empty: удалять пустые строки
    
    Returns:
        list: нормализованные строки
    """

    file_path = filename
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        normalized_lines = []
        for line in lines:
            line = line.strip()
            
            if not line and filter_empty:
                continue
                
            if lowercase:
                line = line.lower()
            
            line = re.sub(r'[^\w\s-]', '', line)  # сохраняем дефисы
            
            if line or not filter_empty:  
                normalized_lines.append(line)
        
        return normalized_lines
    
    except FileNotFoundError:
        print(f"Ошибка: Файл '{file_path}' не найден.")
        return []
    except Exception as e:
        print(f"Произошла ошибка: {e}")
        return []


def main():
    res = rpd_list_generator('rpd_names.txt')
    print(res)


if __name__ == "__main__":
    main()