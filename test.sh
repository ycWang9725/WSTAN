
export CUDA_VISIBLE_DEVICES=3

#python moment_localization/test.py --cfg experiments/charades/2D-TAN-16x16-K5L8-conv-unsup.yaml --verbose --split test
python moment_localization/test.py --cfg experiments/activitynet/2D-TAN-64x64-K9L4-conv-unsup.yaml --verbose --split test