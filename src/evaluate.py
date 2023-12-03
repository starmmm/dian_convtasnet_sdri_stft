#!/usr/bin/env python

# Created on 2018/12
# Author: Kaituo XU

import argparse
import os

import librosa
from mir_eval.separation import bss_eval_sources
from mir_eval.separation import _bss_decomp_mtifilt
import numpy as np
import torch

from data import AudioDataLoader, AudioDataset
from pit_criterion import cal_loss
from conv_tasnet import ConvTasNet
from utils import remove_pad

parser = argparse.ArgumentParser('Evaluate separation performance using Conv-TasNet')
parser.add_argument('--model_path', type=str, required=True,
                    help='Path to model file created by training')
parser.add_argument('--data_dir', type=str, required=True,
                    help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--cal_sdr', type=int, default=0,
                    help='Whether calculate SDR, add this option because calculation of SDR is very slow')
parser.add_argument('--use_cuda', type=int, default=0,
                    help='Whether use GPU')
parser.add_argument('--sample_rate', default=8000, type=int,
                    help='Sample rate')
parser.add_argument('--batch_size', default=1, type=int,
                    help='Batch size')


def evaluate(args):
    total_SISNRi = 0
    total_SDRi = 0
    total_cnt = 0
    total_diff = 0
    count_true = 0

    # Load model
    model = ConvTasNet.load_model(args.model_path)
    print(model)
    model.eval()
    if args.use_cuda:
        model.cuda()

    # Load data
    dataset = AudioDataset(args.data_dir, args.batch_size,
                           sample_rate=args.sample_rate, segment=-1)
    data_loader = AudioDataLoader(dataset, batch_size=1, num_workers=2)

    with torch.no_grad():
        for i, (data) in enumerate(data_loader):
            # Get batch data
            padded_mixture, mixture_lengths, padded_source = data
            if args.use_cuda:
                padded_mixture = padded_mixture.cuda()
                mixture_lengths = mixture_lengths.cuda()
                padded_source = padded_source.cuda()
            # Forward
            estimate_source = model(padded_mixture)  # [B, C, T]
            loss, max_snr, estimate_source, reorder_estimate_source = \
                cal_loss(padded_source, estimate_source, mixture_lengths)
            # Remove padding and flat
            mixture = remove_pad(padded_mixture, mixture_lengths)
            source = remove_pad(padded_source, mixture_lengths)
            # NOTE: use reorder estimate source
            estimate_source = remove_pad(reorder_estimate_source,
                                         mixture_lengths)
            # for each utterance
            for mix, src_ref, src_est in zip(mixture, source, estimate_source):
                print("Utt", total_cnt + 1)
                # Compute SDRi
                if args.cal_sdr:
                    avg_SDRi = cal_SDRi(src_ref, src_est, mix)

                    #计算原始声压级与得到的差值，假设音频为60dB
                    mixture_level  = 60
                    p0 = 2e-5
                    sound_pressure_ratio = np.sqrt((10**(mixture_level/10))*(p0**2)/np.mean(mix**2))
                    mix_presssure = calculate_sound_pressure_level( mix ,sound_pressure_ratio)
                    ref0_presssure = calculate_sound_pressure_level( src_ref[0] ,sound_pressure_ratio)
                    ref1_presssure = calculate_sound_pressure_level( src_ref[1] ,sound_pressure_ratio)
                    est0_presssure = calculate_sound_pressure_level( src_est[0] ,sound_pressure_ratio)
                    est1_presssure = calculate_sound_pressure_level( src_est[1] ,sound_pressure_ratio)
                    print("\tmix_presssure={0:.2f}dB".format(mix_presssure))
                    print("\tref0_presssure={0:.2f}dB".format(ref0_presssure)," est0_presssure={0:.2f}dB".format(est0_presssure)," diff_presssure={0:.2f}dB".format(est0_presssure-ref0_presssure))
                    print("\tref1_presssure={0:.2f}dB".format(ref1_presssure)," est1_presssure={0:.2f}dB".format(est1_presssure)," diff_presssure={0:.2f}dB".format(est1_presssure-ref1_presssure))
                    #
                    diff = (np.abs(est1_presssure-ref1_presssure) + np.abs(est0_presssure-ref0_presssure))/2
                    total_diff +=  diff
                    total_SDRi += avg_SDRi
                    print("\tSDRi={0:.2f}".format(avg_SDRi))
                    if diff<1:
                        count_true += 1
                # Compute SI-SNRi
                avg_SISNRi = cal_SISNRi(src_ref, src_est, mix)
                print("\tSI-SNRi={0:.2f}".format(avg_SISNRi))
                total_SISNRi += avg_SISNRi
                total_cnt += 1
    if args.cal_sdr:
        print("Average SDR improvement: {0:.2f}".format(total_SDRi / total_cnt))
    print("Average SISNR improvement: {0:.2f}".format(total_SISNRi / total_cnt))
    print("Average diff_presssure: {0:.2f}dB".format(total_diff / total_cnt))
    print("声压级差值小于1dB的次数:",count_true,"总次数：", total_cnt )

def calculate_sound_pressure_level(data, ratio):
    data *= ratio
    p0 = 2e-5
    # 计算声压级
    spl = 20 * np.log10(np.sqrt(np.mean(data**2))/p0)

    return spl


def cal_SDRi(src_ref, src_est, mix):
    """Calculate Source-to-Distortion Ratio improvement (SDRi).
    NOTE: bss_eval_sources is very very slow.
    Args:
        src_ref: numpy.ndarray, [C, T]
        src_est: numpy.ndarray, [C, T], reordered by best PIT permutation
        mix: numpy.ndarray, [T]
    Returns:
        average_SDRi
    """
    sisnr1 = cal_SNR(src_ref[0], src_est[0])
    sisnr2 = cal_SNR(src_ref[1], src_est[1])
    sisnr1b = cal_SNR(src_ref[0], mix)
    sisnr2b = cal_SNR(src_ref[1], mix)
    # print("SISNR base1 {0:.2f} SISNR base2 {1:.2f}, avg {2:.2f}".format(
    #     sisnr1b, sisnr2b, (sisnr1b+sisnr2b)/2))
    # print("SISNRi1: {0:.2f}, SISNRi2: {1:.2f}".format(sisnr1, sisnr2))
    avg_SNRi = ((sisnr1 - sisnr1b) + (sisnr2 - sisnr2b)) / 2
    return avg_SNRi

def cal_SNR(ref_sig, out_sig, eps=1e-8):
    """Calcuate Source-to-Noise Ratio (SNR)
    Args:
        ref_sig: numpy.ndarray, [T]
        out_sig: numpy.ndarray, [T]
    Returns:
        SISNR
    """
    assert len(ref_sig) == len(out_sig)
    ref_sig = ref_sig - np.mean(ref_sig)  #s
    out_sig = out_sig - np.mean(out_sig)  #s_est
    s_noise = out_sig - ref_sig

    snr = np.sum(ref_sig ** 2) / (np.sum(s_noise ** 2) + eps)
    snr = 10 * np.log10(snr + eps)  # [B, C, C]
    return snr

# def cal_SDRi(src_ref, src_est, mix):
#     """Calculate Source-to-Distortion Ratio improvement (SDRi).
#     NOTE: bss_eval_sources is very very slow.
#     Args:
#         src_ref: numpy.ndarray, [C, T]
#         src_est: numpy.ndarray, [C, T], reordered by best PIT permutation
#         mix: numpy.ndarray, [T]
#     Returns:
#         average_SDRi
#     """
#     src_anchor = np.stack([mix, mix], axis=0)
#     nsrc = src_est.shape[0]
#     for j in range(nsrc):
#             s_true, e_spat, e_interf, e_artif = \
#                 _bss_decomp_mtifilt(src_ref,
#                                     src_est[j],
#                                     j, 512)
#     sdr, sir, sar, popt = bss_eval_sources(src_ref, src_est)

#     sdr0, sir0, sar0, popt0 = bss_eval_sources(src_ref, src_anchor)
#     avg_SDRi = ((sdr[0]-sdr0[0]) + (sdr[1]-sdr0[1])) / 2
#     # print("SDRi1: {0:.2f}, SDRi2: {1:.2f}".format(sdr[0]-sdr0[0], sdr[1]-sdr0[1]))
#     return avg_SDRi

def cal_SISNRi(src_ref, src_est, mix):
    """Calculate Scale-Invariant Source-to-Noise Ratio improvement (SI-SNRi)
    Args:
        src_ref: numpy.ndarray, [C, T]
        src_est: numpy.ndarray, [C, T], reordered by best PIT permutation
        mix: numpy.ndarray, [T]
    Returns:
        average_SISNRi
    """
    sisnr1 = cal_SISNR(src_ref[0], src_est[0])
    sisnr2 = cal_SISNR(src_ref[1], src_est[1])
    sisnr1b = cal_SISNR(src_ref[0], mix)
    sisnr2b = cal_SISNR(src_ref[1], mix)
    # print("SISNR base1 {0:.2f} SISNR base2 {1:.2f}, avg {2:.2f}".format(
    #     sisnr1b, sisnr2b, (sisnr1b+sisnr2b)/2))
    # print("SISNRi1: {0:.2f}, SISNRi2: {1:.2f}".format(sisnr1, sisnr2))
    avg_SISNRi = ((sisnr1 - sisnr1b) + (sisnr2 - sisnr2b)) / 2
    return avg_SISNRi


def cal_SISNR(ref_sig, out_sig, eps=1e-8):
    """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        ref_sig: numpy.ndarray, [T]
        out_sig: numpy.ndarray, [T]
    Returns:
        SISNR
    """
    assert len(ref_sig) == len(out_sig)
    ref_sig = ref_sig - np.mean(ref_sig)  #s
    out_sig = out_sig - np.mean(out_sig)  #s_est
    ref_energy = np.sum(ref_sig ** 2) + eps #|s|^2
    proj = np.sum(ref_sig * out_sig) * ref_sig / ref_energy # |s*s_est|/|s|^2 *s   水平分量
    noise = out_sig - proj # 垂直分量
    ratio = np.sum(proj ** 2) / (np.sum(noise ** 2) + eps)
    sisnr = 10 * np.log(ratio + eps) / np.log(10.0)
    return sisnr


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    evaluate(args)
