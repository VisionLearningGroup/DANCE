import os
import random
import sys
source = sys.argv[1]
target = sys.argv[2]
p_path = os.path.join('/research/masaito/office/', source,'images')
dir_list = os.listdir(p_path)
#print(dir_list)
class_list_shared = ["back_pack", "bike", "calculator", "headphones", "keyboard", "laptop_computer", "monitor", "mouse", "mug", "projector"]
unshared_list = list(set(dir_list) - set(class_list_shared))
print(class_list_shared)
unshared_list.sort()
#print(unshared_list)
source_list = class_list_shared + unshared_list[:10]
private_t = list(set(unshared_list)- set(source_list))
target_list = class_list_shared  + private_t
print(target_list)
path_source = "../txt/source_%s_obda.txt"%(source)
path_target = "../txt/target_%s_obda.txt"%(target)
write_source = open(path_source,"w")
write_target = open(path_target,"w")
for k, direc in enumerate(dir_list):
    if not '.txt' in direc:
        files = os.listdir(os.path.join(p_path, direc))
        for i, file in enumerate(files):
            if direc in class_list_shared:
                class_name = direc
                file_name = os.path.join('data', source, 'images', direc, file)
                write_source.write('%s %s\n' % (file_name, class_list_shared.index(class_name)))
            else:
                continue
p_path = os.path.join('/research/masaito/office/', target,'images')
dir_list = os.listdir(p_path)
#print(dir_list)
for k, direc in enumerate(target_list):
    if not '.txt' in direc:
        files = os.listdir(os.path.join(p_path, direc))
        for i, file in enumerate(files):
            file_name = os.path.join('data', target, 'images', direc, file)
            if direc in class_list_shared:
                class_name = direc
                write_target.write('%s %s\n' % (file_name, class_list_shared.index(class_name)))
            elif direc in target_list:
                file_name = os.path.join(p_path, direc, file)
                write_target.write('%s %s\n' % (file_name, len(class_list_shared)))


