#!/usr/bin/env bash
python main.py --model densenet --model_config DenseNet121 --train
python plot_contour.py --model densenet --model_config DenseNet121