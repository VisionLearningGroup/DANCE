import sys

file = open(sys.argv[1], 'r')
file_new = open(sys.argv[2], 'w')

files = [line.strip() for line in file.readlines()]

for line in files:
    name = line.split(' ')[0]
    ind = line.split(' ')[1]
    name = name.replace('/research/masaito', 'data')
    file_new.write('%s %s\n'%(name, ind))

