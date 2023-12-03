import argparse
import os
import evaluate

parser = argparse.ArgumentParser('Evaluate separation performance using Conv-TasNet')
if __name__ == '__main__':
    
    expdir="/data01/home/cwc2022/soundSeparate/src/Conv-TasNet/egs/wsj0/exp/all_train_r16000_N256_L20_B256_H512_P3_X8_R4_C2_gLN_causal0_relu_epoch50_half1_norm5_bs6_worker4_adam_lr1e-3_mmt0_l20_tr"
    data="/data01/home/cwc2022/soundSeparate/src/Conv-TasNet/egs/wsj0/data/tt"
    
    args = parser.parse_args()

    # 设置参数的值
    args.model_path = os.path.join(expdir, "final.pth.tar")
    args.data_dir = data
    args.cal_sdr = 1
    args.use_cuda = 1
    args.sample_rate = 48000
    args.batch_size = 2

    # 打印参数
    print(args)

    # 调用 evaluate 函数
    evaluate.evaluate(args)


