import json

def toJson(dir, dict):
    with open(dir, 'w', encoding='utf-8') as file:
        json.dump(dict, file, ensure_ascii=False, indent='\t')

def loadJson(dir):
    with open(dir, 'r', encoding='utf-8') as file:
        return json.load(file)

