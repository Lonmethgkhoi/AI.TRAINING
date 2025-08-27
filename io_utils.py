def read_file(path):
    """Đọc file"""
    with open(path, "r") as f:
        return f.read()

def write_file(path, content):
    """Ghi file"""
    with open(path, "w") as f:
        f.write(content)

def read_json(path):
    """Đọc file JSON"""
    import json
    with open(path, "r") as f:
        return json.load(f)

def write_json(path, data):
    """Ghi file JSON"""
    import json
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def read_csv(path):
    """Đọc file CSV"""
    import csv
    with open(path, newline='') as f:
        return list(csv.reader(f))

def write_csv(path, data):
    """Ghi file CSV"""
    import csv
    with open(path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)
