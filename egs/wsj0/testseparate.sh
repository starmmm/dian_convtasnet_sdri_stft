  #!/bin/bash
  
    #!/bin/bash

# Created on 2018/12
# Author: Kaituo XU

# -- START IMPORTANT
# * If you have mixture wsj0 audio, modify `data` to your path that including tr, cv and tt.
# * If you jsut have origin sphere format wsj0 , modify `wsj0_origin` to your path and
# modify `wsj0_wav` to path that put output wav format wsj0, then read and run stage 1 part.
# After that, modify `data` and run from stage 2.
wsj0_origin=/home/ktxu/workspace/data/CSR-I-WSJ0-LDC93S6A
wsj0_wav=/home/ktxu/workspace/data/wsj0-wav/wsj0
# data=/home/cwc2022/soundSeparate/data/dataset_fs8000/audio
data=/data01/cwc/dataset_mix3/audio
test_audio=/home/cwc2022/soundSeparate/src/Conv-TasNet/mytest_audio
stage=4  # Modify this to control to start from which stage
# -- END

dumpdir=data  # directory to put generated json file

# -- START Conv-TasNet Config
train_dir=$dumpdir/tr
valid_dir=$dumpdir/cv
evaluate_dir=$dumpdir/tt
separate_dir=$dumpdir/tt
sample_rate=8000
segment=3 # seconds
cv_maxlen=6  # seconds
# Network config
N=256
L=20
B=256
H=512
P=3
X=8
R=4
norm_type=gLN
causal=0
mask_nonlinear='relu'
C=2
# Training config
use_cuda=1
id=1
epochs=60
half_lr=1
early_stop=0
max_norm=5
# minibatch
shuffle=1
batch_size=4
num_workers=4
# optimizer
optimizer=adam
lr=1e-3
momentum=0
l2=0
# save and visualize
checkpoint=0
continue_from=""
print_freq=10
visdom=0
visdom_epoch=0
visdom_id="Conv-TasNet Training"
# evaluate
ev_use_cuda=1
cal_sdr=1
# -- END Conv-TasNet Config

# exp tag
tag="" # tag for managing experiments.

ngpu=1  # always 1
separate_out_dir=/home/cwc2022/soundSeparate/src/Conv-TasNet/my_separate_audio
expdir=/home/cwc2022/soundSeparate/src/Conv-TasNet/egs/wsj0/exp/train_r8000_N256_L20_B256_H512_P3_X8_R4_C2_gLN_causal0_relu_epoch10_half1_norm5_bs4_worker4_adam_lr1e-3_mmt0_l20_tr
separate_dir=/home/cwc2022/soundSeparate/src/Conv-TasNet/mytest_audio
separate_out_dir=/home/cwc2022/soundSeparate/src/Conv-TasNet/my_separate_audio 
test_audio=/home/cwc2022/soundSeparate/src/Conv-TasNet/mytest_audio
ev_use_cuda=1
sample_rate=8000
batch_size=4

. utils/parse_options.sh || exit 1;
. ./cmd.sh
. ./path.sh

echo "Stage 4: Separate speech using Conv-TasNet"



  ${decode_cmd} --gpu ${ngpu} ${separate_out_dir}/separate.log \
    separate.py \
    --model_path ${expdir}/final.pth.tar \
    --mix_json $separate_dir/mix.json \
    --out_dir ${separate_out_dir} \
    --mix_dir $test_audio \
    --use_cuda $ev_use_cuda \
    --sample_rate $sample_rate \
    --batch_size $batch_size