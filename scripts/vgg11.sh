#!/usr/bin/env bash
python main.py --model vgg --model_config VGG11 --train
python plot_contour.py --model vgg --model_config VGG11