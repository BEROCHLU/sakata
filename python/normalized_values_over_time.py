import os
import re
import datetime
import matplotlib.pyplot as plt

file_paths = ["./result/nbatch.log", "./result/sakata-batch.log"]
norm_values_list = []
date_pattern = re.compile(r"\d{4}-\d{2}-\d{2}")

for file_path in file_paths:
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    sections = content.split("\n\n")
    dates = []
    norm_values = []

    for section in sections:
        lines = section.split("\n")
        lines = lines[1:]

        for line in reversed(lines):
            date_match = date_pattern.search(line)
            if date_match:
                date_str = date_match.group()
                date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
                dates.append(date)
                break

        norm_line = [line for line in lines if line.startswith("Norm:")]
        if norm_line:
            norm_value = float(norm_line[0].split(":")[1].strip())
            norm_values.append(norm_value)

    norm_values_list.append(norm_values)

plt.figure(figsize=(10, 6))
for norm_values in norm_values_list:
    plt.plot(dates, norm_values, marker="o")
plt.xlabel("Date")
plt.ylabel("Norm: value")
plt.title("Norm: value over time")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("./result/norm_value_by_date.png")
plt.show()
