import os

l = []

# Python is zero-index based
for i in range(1, 101):
    l.append(f"Folder_{i}")

for folder in l:
    os.makedirs(folder)
