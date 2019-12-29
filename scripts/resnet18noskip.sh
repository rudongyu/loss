#!/usr/bin/env bash
python main.py --model resnet --model_config ResNet18NoSkip --train
python plot_contour.py --model resnet --model_config ResNet18NoSkip