#!/bin/sh
python train_dance.py --config configs/officehome-train-config_PDA.yaml --source ./txt/source_Real_pada.txt --target ./txt/target_Art_pada.txt --gpu $1
python train_dance.py --config configs/officehome-train-config_PDA.yaml --source ./txt/source_Real_pada.txt --target ./txt/target_Clipart_pada.txt --gpu $1
python train_dance.py --config configs/officehome-train-config_PDA.yaml --source ./txt/source_Real_pada.txt --target ./txt/target_Product_pada.txt --gpu $1
python train_dance.py --config configs/officehome-train-config_PDA.yaml --source ./txt/source_Product_pada.txt --target ./txt/target_Real_pada.txt --gpu $1
python train_dance.py --config configs/officehome-train-config_PDA.yaml --source ./txt/source_Product_pada.txt --target ./txt/target_Art_pada.txt --gpu $1
python train_dance.py --config configs/officehome-train-config_PDA.yaml --source ./txt/source_Product_pada.txt --target ./txt/target_Clipart_pada.txt --gpu $1
python train_dance.py --config configs/officehome-train-config_PDA.yaml --source ./txt/source_Art_pada.txt --target ./txt/target_Clipart_pada.txt --gpu $1
python train_dance.py --config configs/officehome-train-config_PDA.yaml --source ./txt/source_Art_pada.txt --target ./txt/target_Product_pada.txt --gpu $1
python train_dance.py --config configs/officehome-train-config_PDA.yaml --source ./txt/source_Art_pada.txt --target ./txt/target_Real_pada.txt --gpu $1
python train_dance.py --config configs/officehome-train-config_PDA.yaml --source ./txt/source_Clipart_pada.txt --target ./txt/target_Real_pada.txt --gpu $1
python train_dance.py --config configs/officehome-train-config_PDA.yaml --source ./txt/source_Clipart_pada.txt --target ./txt/target_Product_pada.txt --gpu $1
python train_dance.py --config configs/officehome-train-config_PDA.yaml --source ./txt/source_Clipart_pada.txt --target ./txt/target_Art_pada.txt --gpu $1
