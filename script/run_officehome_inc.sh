#!/bin/sh
python train_class_inc_dance.py --config configs/officehome-train-config_CLDA.yaml --exp_name $3 --source ./txt/source_Real_obda.txt --target_u ./txt/target_Clipart_cls.txt --target_l ./txt/target_Clipart_labeled.txt --gpu_device $1
python train_class_inc_dance.py --config configs/officehome-train-config_CLDA.yaml --exp_name $3 --source ./txt/source_Real_obda.txt --target_u ./txt/target_Art_cls.txt --target_l ./txt/target_Art_labeled.txt --gpu_device $1
python train_class_inc_dance.py --config configs/officehome-train-config_CLDA.yaml --exp_name $3 --source ./txt/source_Real_obda.txt --target_u ./txt/target_Product_cls.txt --target_l ./txt/target_Product_labeled.txt --gpu_device $1
python train_class_inc_dance.py --config configs/officehome-train-config_CLDA.yaml --exp_name $3 --source ./txt/source_Product_obda.txt --target_u ./txt/target_Real_cls.txt --target_l ./txt/target_Real_labeled.txt --gpu_device $1
python train_class_inc_dance.py --config configs/officehome-train-config_CLDA.yaml --exp_name $3 --source ./txt/source_Product_obda.txt --target_u ./txt/target_Art_cls.txt --target_l ./txt/target_Art_labeled.txt --gpu_device $1
python train_class_inc_dance.py --config configs/officehome-train-config_CLDA.yaml --exp_name $3 --source ./txt/source_Product_obda.txt --target_u ./txt/target_Clipart_cls.txt --target_l ./txt/target_Clipart_labeled.txt --gpu_device $1
python train_class_inc_dance.py --config configs/officehome-train-config_CLDA.yaml --exp_name $3 --source ./txt/source_Art_obda.txt --target_u ./txt/target_Clipart_cls.txt --target_l ./txt/target_Clipart_labeled.txt --gpu_device $1
python train_class_inc_dance.py --config configs/officehome-train-config_CLDA.yaml --exp_name $3 --source ./txt/source_Art_obda.txt --target_u ./txt/target_Product_cls.txt --target_l ./txt/target_Product_labeled.txt --gpu_device $1
python train_class_inc_dance.py --config configs/officehome-train-config_CLDA.yaml --exp_name $3 --source ./txt/source_Art_obda.txt --target_u ./txt/target_Real_cls.txt --target_l ./txt/target_Real_labeled.txt --gpu_device $1
python train_class_inc_dance.py --config configs/officehome-train-config_CLDA.yaml --exp_name $3 --source ./txt/source_Clipart_obda.txt --target_u ./txt/target_Real_cls.txt --target_l ./txt/target_Real_labeled.txt --gpu_device $1
python train_class_inc_dance.py --config configs/officehome-train-config_CLDA.yaml --exp_name $3 --source ./txt/source_Clipart_obda.txt --target_u ./txt/target_Art_cls.txt --target_l ./txt/target_Art_labeled.txt --gpu_device $1
python train_class_inc_dance.py --config configs/officehome-train-config_CLDA.yaml --exp_name $3 --source ./txt/source_Clipart_obda.txt --target_u ./txt/target_Product_cls.txt --target_l ./txt/target_Product_labeled.txt --gpu_device $1

