#!/bin/sh
python train_dance.py --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Real_obda.txt --target ./txt/target_Art_obda.txt --gpu $1
python train_dance.py --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Real_obda.txt --target ./txt/target_Clipart_obda.txt --gpu $1
python train_dance.py --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Real_obda.txt --target ./txt/target_Product_obda.txt --gpu $1
python train_dance.py --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Product_obda.txt --target ./txt/target_Real_obda.txt --gpu $1
python train_dance.py --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Product_obda.txt --target ./txt/target_Art_obda.txt --gpu $1
python train_dance.py --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Product_obda.txt --target ./txt/target_Clipart_obda.txt --gpu $1
python train_dance.py --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Art_obda.txt --target ./txt/target_Clipart_obda.txt --gpu $1
python train_dance.py --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Art_obda.txt --target ./txt/target_Product_obda.txt --gpu $1
python train_dance.py --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Art_obda.txt --target ./txt/target_Real_obda.txt --gpu $1
python train_dance.py --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Clipart_obda.txt --target ./txt/target_Real_obda.txt --gpu $1
python train_dance.py --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Clipart_obda.txt --target ./txt/target_Product_obda.txt --gpu $1
python train_dance.py --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Clipart_obda.txt --target ./txt/target_Art_obda.txt --gpu $1
