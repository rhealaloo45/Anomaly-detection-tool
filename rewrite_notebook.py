import json

with open('isolation.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'log_pattern = re.compile' in ''.join(cell['source']):
        cell['source'] = [
            \
