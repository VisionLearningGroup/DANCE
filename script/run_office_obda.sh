#!/bin/sh
python train_dance.py --config $2 --source ./txt/source_amazon_obda.txt --target ./txt/target_dslr_obda.txt --gpu $1
python train_dance.py --config $2 --source ./txt/source_amazon_obda.txt --target ./txt/target_webcam_obda.txt --gpu $1
python train_dance.py --config $2 --source ./txt/source_dslr_obda.txt --target ./txt/target_webcam_obda.txt --gpu $1
python train_dance.py --config $2 --source ./txt/source_dslr_obda.txt --target ./txt/target_amazon_obda.txt --gpu $1
python train_dance.py --config $2 --source ./txt/source_webcam_obda.txt --target ./txt/target_amazon_obda.txt --gpu $1
python train_dance.py --config $2 --source ./txt/source_webcam_obda.txt --target ./txt/target_dslr_obda.txt --gpu $1