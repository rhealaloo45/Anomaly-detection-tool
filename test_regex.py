import csv
import re
f=open('Masked_log1.csv', 'r', encoding='utf-8')
reader=csv.reader(f)
row1=next(reader)[0]
row2=next(reader)[0]
row3=next(reader)[0]
print(repr(row1))
print(repr(row2))
print(repr(row3))
pat = re.compile(r'<150>(\w{3}\s+\d+\s+\d{2}:\d{2}:\d{2})\s+(\S+)\s+(.*?):\s+(.*?)\s+-\s+-\s+\[(.*?)\]\s+\"(.*?)\"\s+(\d+)\s+(\S+)\s+(\d+)(?:\s+\d+)?\s+\"(.*?)\"\s+\"(.*?)\"')
print(pat.match(row1).groups())
print(pat.match(row3).groups())
