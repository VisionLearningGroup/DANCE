import os
import sys
source = sys.argv[1]
target = sys.argv[2]
p_path = os.path.join('/research/masaito/OfficeHomeDataset_10072016/', source)
dir_list = os.listdir(p_path)
dir_list.sort()

source_list = dir_list[:15]
target_list = dir_list
print(source_list)
print(target_list)
path_source = "../txt/source_%s_obda.txt"%(source)
path_target = "../txt/target_%s_obda.txt"%(target)
write_source = open(path_source,"w")
write_target = open(path_target,"w")

for k, direc in enumerate(source_list):
    if not '.txt' in direc:
        files = os.listdir(os.path.join(p_path, direc))
        for i, file in enumerate(files):
            if direc in source_list:
                class_name = direc
                file_name = os.path.join('data', source, direc, file)
                write_source.write('%s %s\n' % (file_name, source_list.index(class_name)))
            else:
                continue

p_path = os.path.join('/research/masaito/OfficeHomeDataset_10072016/', target)
dir_list = os.listdir(p_path)
dir_list.sort()
for k, direc in enumerate(target_list):
    if not '.txt' in direc:
        files = os.listdir(os.path.join(p_path, direc))
        for i, file in enumerate(files):
            file_name = os.path.join('data', target, direc, file)
            if direc in source_list:
                class_name = direc
                write_target.write('%s %s\n' % (file_name, source_list.index(class_name)))
            elif direc in target_list:
                write_target.write('%s %s\n' % (file_name, len(source_list)))


