#cd /gdata1/wangyc/2D-TAN
#export CUDA_VISIBLE_DEVICES=4,5,6,7

python moment_localization/train_squeeze.py --cfg experiments/squeeze/2D-TAN-16x16-K5L8-conv-unsup.yaml --verbose
#python moment_localization/train.py --cfg experiments/charades/2D-TAN-16x16-K5L8-conv-unsup.yaml --verbose

#experiments/didemo/2D-TAN-6x6-K3L8-conv-unsup.yaml
#experiments/charades/2D-TAN-16x16-K5L8-conv-unsup.yaml
#experiments/activitynet/2D-TAN-64x64-K9L4-conv-unsup.yaml

#squeeze/2D-TAN-16x16-K5L8-conv-unsup.yaml

#python moment_localization/test.py --cfg experiments/activitynet/2D-TAN-64x64-K9L4-conv-unsup.yaml --verbose --split test