import json

def split_dataset(input_file='datasets/dataset.json'):
    """
    Разделяет dataset на три файла по router_type
    """
    # Загрузка данных
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Разделение по категориям
    single = []
    relation = []
    global_ds = []
    
    for item in data:
        router_type = item.get('router_type', '')
        if router_type in ['single.simple', 'single.global']:
            single.append(item)
        elif router_type == 'multi.relation':
            relation.append(item)
        elif router_type == 'multi.global':
            global_ds.append(item)
    
    # Сохранение файлов в той же папке
    output_dir = '/'.join(input_file.split('/')[:-1]) or '.'
    
    with open(f'{output_dir}/single_ds.json', 'w', encoding='utf-8') as f:
        json.dump(single, f, ensure_ascii=False, indent=2)
    
    with open(f'{output_dir}/relation_ds.json', 'w', encoding='utf-8') as f:
        json.dump(relation, f, ensure_ascii=False, indent=2)
    
    with open(f'{output_dir}/global_ds.json', 'w', encoding='utf-8') as f:
        json.dump(global_ds, f, ensure_ascii=False, indent=2)
    
    print(f"Готово! single_ds.json: {len(single)}, relation_ds.json: {len(relation)}, global_ds.json: {len(global_ds)}")

# Использование
if __name__ == '__main__':
    split_dataset('datasets/dataset.json')