#!/usr/bin/env bash
python main.py --model vgg --model_config VGG16 --train
python plot_contour.py --model vgg --model_config VGG16