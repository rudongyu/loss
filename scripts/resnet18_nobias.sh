#!/usr/bin/env bash
python main.py --model resnet --model_config ResNet18 --size 50 --out_dir outputnobias
python plot_contour.py --model resnet --model_config ResNet18 --size 50 --out_dir outputnobias