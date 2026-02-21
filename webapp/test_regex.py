import re
file_path = 'webapp/uploads/test_example.log'

p1 = r'^<150>([A-Za-z]{3}\s+\d+\s+\d{2}:\d{2}:\d{2})\s+(\S+)\s+(.*?):\s+(.*?)\s+-\s+-\s+\[(.*?)\]\s+"(.*?)"\s+(\d+)\s+(\S+)\s+(\S+)\s+\d*\s*"(.*?)"\s+"(.*?)"'

with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        m = re.match(p1, line.strip())
        if m:
            print("MATCH:", m.groups()[3])  # checking IP block
        else:
            print("FAIL:", line[:50])
