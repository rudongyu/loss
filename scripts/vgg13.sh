#!/usr/bin/env bash
python main.py --model vgg --model_config VGG13 --train
python plot_contour.py --model vgg --model_config VGG13