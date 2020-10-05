# [Universal Domain Adaptation through Self-Supervision (NeurlPS 2020)](https://arxiv.org/pdf/2002.07953.pd)

This repository provides code for the paper, Universal Domain Adaptation through Self-Supervision.
Please go to our project page to quickly understand the content of the paper or read our paper.
### [Project Page](http://cs-people.bu.edu/keisaito/research/DANCE.html)  [Paper (will be updated soon)](https://arxiv.org/pdf/2002.07953.pdf)


## Environment
Python 3.6.9, Pytorch 1.2.0, Torch Vision 0.4, [Apex](https://github.com/NVIDIA/apex). See requirement.txt.
 We used the nvidia apex library for memory efficient high-speed training.

## Data Preparation

[Office Dataset](https://people.eecs.berkeley.edu/~jhoffman/domainadapt/)
[OfficeHome Dataset](http://hemanthdv.org/OfficeHome-Dataset/) [VisDA](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification)

Prepare dataset in data directory as follows.
```
./data/amazon/images/ ## Office
./data/Real/ ## OfficeHome
./data/visda_train/ ## VisDA synthetic images
./data/visda_val/ ## VisDA real images
```
Prepare image list.
```
unzip txt.zip
```
File list has to be stored in ./txt.


## Train

All training script is stored in script directory.

Example: Open Set Domain Adaptation on Office.
```
sh script/run_office_obda.sh $gpu-id configs/office-train-config_ODA.yaml
```

### Reference
This repository is contributed by [Kuniaki Saito](http://cs-people.bu.edu/keisaito/).
If you consider using this code or its derivatives, please consider citing:

```
@inproceedings{saito2020dance,
  title={Universal Domain Adaptation through Self-Supervision},
  author={Saito, Kuniaki and Kim, Donghyun and Sclaroff, Stan and Saenko, Kate},
  journal={NeurIPS},
  year={2020}
}
```

