import csv
import json
import io
import pyperclip

def process_csv_input(csv_input):
    lines = csv_input.strip().splitlines()
    valid_lines = []
    consecutive_newlines = 0
    for line in lines:
        if line.strip():
            valid_lines.append(line)
            consecutive_newlines = 0
        else:
            consecutive_newlines += 1
            if consecutive_newlines == 2:
                break
    if not valid_lines:
        return None
    csv_data = '\n'.join(valid_lines)
    csv_file = io.StringIO(csv_data)
    reader = csv.DictReader(csv_file)
    json_data = []
    for row in reader:
        json_row = {key: (value if value else None) for key, value in row.items()}
        json_data.append(json_row)
    return json_data
print("Enter the CSV data:")
csv_input = ''
while True:
    line = input()
    if line == '' and csv_input.endswith('\n'):
        break
    csv_input += line + '\n'
json_data = process_csv_input(csv_input)
if json_data:
    json_output = json.dumps(json_data, indent=4)
    print("Converted JSON:")
    print(json_output)
    pyperclip.copy(json_output)
    print("JSON has been copied to clipboard.")
else:
    print("Invalid CSV.")