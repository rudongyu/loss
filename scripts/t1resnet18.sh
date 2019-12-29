#!/usr/bin/env bash
python main.py --model resnet --model_config ResNet18 --train --size 20 --train_ratio 0.01 --out_dir out0.01 --epoch_num 1000 --lr_decay 0.95
python plot_contour.py --model resnet --model_config ResNet18 --size 20 --out_dir out0.01