#!/bin/sh
python train_dance.py --config configs/officehome-train-config_CDA.yaml --source ./txt/source_Real_cls.txt --target ./txt/target_Art_cls.txt --gpu $1
python train_dance.py --config configs/officehome-train-config_CDA.yaml --source ./txt/source_Real_cls.txt --target ./txt/target_Clipart_cls.txt --gpu $1
python train_dance.py --config configs/officehome-train-config_CDA.yaml --source ./txt/source_Real_cls.txt --target ./txt/target_Product_cls.txt --gpu $1
python train_dance.py --config configs/officehome-train-config_CDA.yaml --source ./txt/source_Product_cls.txt --target ./txt/target_Real_cls.txt --gpu $1
python train_dance.py --config configs/officehome-train-config_CDA.yaml --source ./txt/source_Product_cls.txt --target ./txt/target_Art_cls.txt --gpu $1
python train_dance.py --config configs/officehome-train-config_CDA.yaml --source ./txt/source_Product_cls.txt --target ./txt/target_Clipart_cls.txt --gpu $1
python train_dance.py --config configs/officehome-train-config_CDA.yaml --source ./txt/source_Art_cls.txt --target ./txt/target_Clipart_cls.txt --gpu $1
python train_dance.py --config configs/officehome-train-config_CDA.yaml --source ./txt/source_Art_cls.txt --target ./txt/target_Product_cls.txt --gpu $1
python train_dance.py --config configs/officehome-train-config_CDA.yaml --source ./txt/source_Art_cls.txt --target ./txt/target_Real_cls.txt --gpu $1
python train_dance.py --config configs/officehome-train-config_CDA.yaml --source ./txt/source_Clipart_cls.txt --target ./txt/target_Real_cls.txt --gpu $1
python train_dance.py --config configs/officehome-train-config_CDA.yaml --source ./txt/source_Clipart_cls.txt --target ./txt/target_Product_cls.txt --gpu $1
python train_dance.py --config configs/officehome-train-config_CDA.yaml --source ./txt/source_Clipart_cls.txt --target ./txt/target_Art_cls.txt --gpu $1
