import os
import random
from pydub import AudioSegment

def get_random_audio(file_path):
    audio_files = [f for f in os.listdir(file_path) if f.endswith('.mp3') or f.endswith('.wav')]
    if not audio_files:
        return None
    selected_file = random.choice(audio_files)
    audio = AudioSegment.from_file(os.path.join(file_path, selected_file))
    return audio

def mix_and_cut_audio(folder_paths, output_folder, num_mixes, cut_duration=3000):
    for i in range(num_mixes):
        audios = [get_random_audio(path) for path in folder_paths]

        if None in audios:
            print("Error: One or more folders do not contain compatible audio files.")
            return

        # 剪切前3秒
        audios = [audio[:cut_duration] for audio in audios]

        mixed_audio = audios[0]

        for audio in audios[1:]:
            snr = random.uniform(-5, 5)  # 随机生成信噪比
            # 计算混合时的音量调整
            target_rms = mixed_audio.dBFS - snr
            adjustment_needed = target_rms - audio.dBFS
            adjusted_audio = audio + adjustment_needed

            # 混合音频
            mixed_audio = mixed_audio.overlay(adjusted_audio)

        output_path = os.path.join(output_folder, f"mixed_audio_{i + 1}.wav")
        mixed_audio.export(output_path, format="wav")
        print(f"Mixing complete. Mixed and cut audio saved to {output_path}")


if __name__ == "__main__":
    folder_paths = ["/data01/cwc/data_fs16000/transformer_add", "/data01/cwc/data_fs16000/background/birds_resample_16000", "/data01/cwc/data_fs16000/background/human_voice"]
    output_folder = "/data01/home/cwc2022/soundSeparate/src/Conv-TasNet/mytest_audio"
    num_mixes = 10

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    mix_and_cut_audio(folder_paths, output_folder, num_mixes)
