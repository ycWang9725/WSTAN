
export CUDA_VISIBLE_DEVICES=2

python moment_localization/train.py --cfg experiments/charades/2D-TAN-16x16-K5L8-conv-unsup.yaml --verbose
