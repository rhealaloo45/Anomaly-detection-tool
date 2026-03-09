import csv
import re
import pandas as pd

def parse_log_line(line):
    match = re.match(r'<150>(\w{3}\s+\d+\s+\d{2}:\d{2}:\d{2})\s+(\S+)\s+(.*?):\s+(.*?)\s+(?:-\s+)?-\s+\[(.*?)\]\s+\"(.*?)\"\s+(\d+)\s+(\S+)(.*)', line)
    if not match: return None
    return True

data = []
with open('Masked_log1.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    fails = 0
    for i, row in enumerate(reader):
        if not row: continue
        parsed = parse_log_line(row[0].strip())
        if not parsed:
            fails += 1
            if fails <= 3:
                print('Failed:', repr(row[0]))
print(f'Failed {fails}')
