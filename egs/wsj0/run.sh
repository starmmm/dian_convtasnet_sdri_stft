#!/bin/bash

# Created on 2018/12
# Author: Kaituo XU

# -- START IMPORTANT
# * If you have mixture wsj0 audio, modify `data` to your path that including tr, cv and tt.
# * If you jsut have origin sphere format wsj0 , modify `wsj0_origin` to your path and
# modify `wsj0_wav` to path that put output wav format wsj0, then read and run stage 1 part.
# After that, modify `data` and run from stage 2.
# wsj0_origin=/home/ktxu/workspace/data/CSR-I-WSJ0-LDC93S6A
# wsj0_wav=/home/ktxu/workspace/data/wsj0-wav/wsj0
# data=/home/cwc2022/soundSeparate/data/dataset_fs8000/audio
# data=/data01/cwc/dataset/data2000_fs16000_mix_bird_insects/audio 
name=stft_environment_and_insubstation_loss_SDR
data=/data01/cwc/dataset_fs48000/s1_3environment_s2_mix_corana_transformer_snr5_num6000/audio
test_audio=/data01/home/cwc2022/soundSeparate/src/Conv-TasNet/mytest_audio
stage=3  # Modify this to control to start from which stage
# -- END

dumpdir=data  # directory to put generated json file

# -- START Conv-TasNet Config
train_dir=$dumpdir/tr
valid_dir=$dumpdir/cv
evaluate_dir=$dumpdir/tt
separate_dir=$dumpdir/tt
sample_rate=48000
segment=3 # seconds
cv_maxlen=3  # seconds
# Network config
N=512
L=40
B=128
H=256
P=3   #Kernel size in convolutional blocks
X=10  #Number of convolutional blocks in each repeat
R=4   #Number of repeats
norm_type=gLN
causal=0
mask_nonlinear='relu'
C=2
# Training config
use_cuda=1
id=2
epochs=10
half_lr=1
early_stop=1
max_norm=5
# minibatch
shuffle=1
batch_size=2
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

. utils/parse_options.sh || exit 1;
. ./cmd.sh
. ./path.sh


if [ $stage -le 0 ]; then
  echo "Stage 0: Convert sphere format to wav format and generate mixture"
  local/data_prepare.sh --data ${wsj0_origin} --wav_dir ${wsj0_wav}

  echo "NOTE: You should generate mixture by yourself now.
You can use tools/create-speaker-mixtures.zip which is download from
http://www.merl.com/demos/deep-clustering/create-speaker-mixtures.zip
If you don't have Matlab and want to use Octave, I suggest to replace
all mkdir(...) in create_wav_2speakers.m with system(['mkdir -p '...])
due to mkdir in Octave can not work in 'mkdir -p' way.
e.g.:
mkdir([output_dir16k '/' min_max{i_mm} '/' data_type{i_type}]);
->
system(['mkdir -p ' output_dir16k '/' min_max{i_mm} '/' data_type{i_type}]);"
  exit 1
fi


if [ $stage -le 1 ]; then
  echo "Stage 1: Generating json files including wav path and duration"
  [ ! -d $dumpdir ] && mkdir $dumpdir
  preprocess.py --in-dir $data --out-dir $dumpdir --sample-rate $sample_rate
fi


if [ -z ${tag} ]; then
  expdir=exp/${name}_r${sample_rate}_N${N}_L${L}_B${B}_H${H}_P${P}_X${X}_R${R}_C${C}_${norm_type}_causal${causal}_${mask_nonlinear}_epoch${epochs}_half${half_lr}_norm${max_norm}_bs${batch_size}_worker${num_workers}_${optimizer}_lr${lr}_mmt${momentum}_l2${l2}_`basename $train_dir`
else
  expdir=exp/train_${tag}
fi

if [ $stage -le 2 ]; then
  echo "Stage 2: Training"
  ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
    CUDA_VISIBLE_DEVICES="$id" \
    train.py \
    --train_dir $train_dir \
    --valid_dir $valid_dir \
    --sample_rate $sample_rate \
    --segment $segment \
    --cv_maxlen $cv_maxlen \
    --N $N \
    --L $L \
    --B $B \
    --H $H \
    --P $P \
    --X $X \
    --R $R \
    --C $C \
    --norm_type $norm_type \
    --causal $causal \
    --mask_nonlinear $mask_nonlinear \
    --use_cuda $use_cuda \
    --epochs $epochs \
    --half_lr $half_lr \
    --early_stop $early_stop \
    --max_norm $max_norm \
    --shuffle $shuffle \
    --batch_size $batch_size \
    --num_workers $num_workers \
    --optimizer $optimizer \
    --lr $lr \
    --momentum $momentum \
    --l2 $l2 \
    --save_folder ${expdir} \
    --checkpoint $checkpoint \
    --continue_from "$continue_from" \
    --print_freq ${print_freq} \
    --visdom $visdom \
    --visdom_epoch $visdom_epoch \
    --visdom_id "$visdom_id"
fi

# expdir=/data01/home/cwc2022/soundSeparate/src/Conv-TasNet/egs/wsj0/exp/continue_from_r48000_N512_L40_B128_H256_P3_X7_R2_C2_gLN_causal0_relu_epoch50_half1_norm5_bs2_worker4_adam_lr1e-3_mmt0_l20_tr
# expdir=/data01/home/cwc2022/soundSeparate/src/Conv-TasNet/egs/wsj0/exp/all_train_r16000_N256_L20_B256_H512_P3_X8_R4_C2_gLN_causal0_relu_epoch50_half1_norm5_bs6_worker4_adam_lr1e-3_mmt0_l20_tr
if [ $stage -le 3 ]; then
  echo "Stage 3: Evaluate separation performance"
  ${decode_cmd} --gpu ${ngpu} ${expdir}/evaluate.log \
    CUDA_VISIBLE_DEVICES="$id" \
    evaluate.py \
    --model_path ${expdir}/final.pth.tar \
    --data_dir $evaluate_dir \
    --cal_sdr $cal_sdr \
    --use_cuda $ev_use_cuda \
    --sample_rate $sample_rate \
    --batch_size $batch_size
fi


if [ $stage -le 4 ]; then
  echo "Stage 4: Separate speech using Conv-TasNet"
  separate_out_dir=${expdir}/separate
  ${decode_cmd} --gpu ${ngpu} ${separate_out_dir}/separate.log \
    CUDA_VISIBLE_DEVICES="$id" \
    separate.py \
    --model_path ${expdir}/final.pth.tar \
    --mix_json $separate_dir/mix.json \
    --out_dir ${separate_out_dir} \
    --mix_dir $test_audio \
    --use_cuda $ev_use_cuda \
    --sample_rate $sample_rate \
    --batch_size $batch_size

fi
