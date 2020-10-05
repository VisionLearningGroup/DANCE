#!/bin/sh
python train_dance.py --config configs/visda-train-config_ODA.yaml --source ./txt/source_visda.txt --target ./txt/target_visda.txt --gpu $1
python train_dance.py --config configs/visda-train-config_UDA.yaml --source ./txt/source_visda_univ.txt --target ./txt/target_visda_univ.txt --gpu $1
python train_dance.py --config configs/visda-train-config_PDA.yaml --source ./txt/source_visda_pada.txt --target ./txt/target_visda_pada.txt --gpu $1
python train_dance.py --config configs/visda-train-config_CDA.yaml --source ./txt/source_visda_cls.txt --target ./txt/target_visda_cls.txt --gpu $1
