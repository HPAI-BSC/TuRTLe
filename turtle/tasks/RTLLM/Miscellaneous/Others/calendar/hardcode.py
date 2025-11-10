import os
import sys

output = []
with open("./RTLLM/Miscellaneous/Others/calendar/reference.txt", "r") as fd:
    for i, line in enumerate(fd):
        output.append(f"reference_data[{i}] = 18'h{line.strip()};")

with open("./RTLLM/Miscellaneous/Others/calendar/hardcoded.txt", "w") as fd:
    for line in output:
        fd.write(f"{line}\n")
