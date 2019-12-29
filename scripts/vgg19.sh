#!/usr/bin/env bash
python main.py --model vgg --model_config VGG19 --train
python plot_contour.py --model vgg --model_config VGG19