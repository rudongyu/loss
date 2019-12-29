#!/usr/bin/env bash
python main.py --model resnet --model_config ResNet34NoSkip --train
python plot_contour.py --model resnet --model_config ResNet34NoSkip