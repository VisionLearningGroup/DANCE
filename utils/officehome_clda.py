import os
import sys
import random
source = sys.argv[1]
target = sys.argv[2]
p_path = os.path.join('/research/masaito/OfficeHomeDataset_10072016/', source)
dir_list = os.listdir(p_path)
dir_list.sort()

source_list = dir_list[:15]
target_list = dir_list
print(source_list)
print(target_list)
path_target = "../txt/target_%s_labeled.txt"%(target)
write_target = open(path_target,"w")

p_path = os.path.join('/research/masaito/OfficeHomeDataset_10072016/', target)
dir_list = os.listdir(p_path)
dir_list.sort()
count = 0
for k, direc in enumerate(target_list):
    if not '.txt' in direc:
        files = os.listdir(os.path.join(p_path, direc))
        random.shuffle(files)
        file_name = os.path.join('data', target, direc, files[0])
        write_target.write('%s %s\n' % (file_name, dir_list.index(direc)))



