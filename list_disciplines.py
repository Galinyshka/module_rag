import json
from retrieval import RetrievalModule

if __name__ == "__main__":
    retrieval = RetrievalModule()
    names = list(retrieval._discipline_index)
    for name in names:
        print(name)
    # Сохраняем в файл для дальнейшего использования в config.py
    with open("rpd_names.json", "w", encoding="utf-8") as f:
        json.dump(names, f, ensure_ascii=False, indent=2)