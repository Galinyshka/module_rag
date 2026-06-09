import json
from rag.retrieval.retrieval import RetrievalModule

RPD_NAMES_PATH = 'src/rag/utils/rpd_names.json'

if __name__ == "__main__":
    retrieval = RetrievalModule()
    names = list(retrieval._discipline_index)
    for name in names:
        print(name)
    # Сохраняем в файл для дальнейшего использования в config.py
    with open(RPD_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(names, f, ensure_ascii=False, indent=2)