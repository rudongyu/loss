#!/usr/bin/env bash
python main.py --model densenet --model_config DenseNet161 --train
python plot_contour.py --model densenet --model_config DenseNet161