import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
import argparse
import scipy.signal
import soundfile as sf
import norbert
import json
from pathlib import Path
import utils
import tqdm
import io
import time
import matplotlib.pyplot as plt
import importlib
from u2net import u2net

def load_model(target, model_name='umxhq', device=torch.device("cpu")):
    model_path = Path(model_name).expanduser()
    if not model_path.exists():
        print("模型不存在")
    else:
        with open(Path(model_path, target + '.json'), 'r') as stream:
            results = json.load(stream)

        target_model_path = next(Path(model_path).glob("%s.pth" % target))
        # print(target_model_path)
        state = torch.load(target_model_path,map_location=device)

        mymodel = u2net(2,2,results['args']['bins'])
        
        mymodel.load_state_dict(state)
        mymodel.eval()
        mymodel.to(device)

        params = {
            'fft': results['args']['fft'],
            'hop': results['args']['hop'],
            'dur': results['args']['dur'],
            'channels': results['args']['channels'],
            'sample_rate': results['args']['sample_rate'],

        }
        return mymodel, params


def transform(audio, model, fft, hop, device):
    with torch.no_grad():

        audio_torch = utils.Spectrogram(utils.STFT(audio[None, ...], None, fft, hop))
        audio_torch = audio_torch.to(device)
        mag_target = model(audio_torch)

        mag_target, mag_mask = model(audio_torch)
        mag_target = mag_target * F.sigmoid(mag_mask)
        mag_target = mag_target.cpu().detach()
        
        mag_target = mag_target.reshape(-1, mag_target.shape[-2], mag_target.shape[-1])
        X = torch.stft(audio, fft, hop, window=torch.hann_window(fft))
        magnitude, phase = torchaudio.functional.magphase(X)
        complex = torch.stack((mag_target * torch.cos(phase), mag_target * torch.sin(phase)), -1)
        audio_hat = torch.istft(complex, fft, hop, fft, torch.hann_window(fft)).numpy()

    return audio_hat


def separate(
        input_file, target, model_name='umxhq', device=torch.device("cpu")
):

    Model, params = load_model(target=target, model_name=model_name, device=device)

    fft = params['fft']
    hop = params['hop']
    dur = params['dur']

    channels = params['channels']
    sample_rate = params['sample_rate']

    audio, rate = torchaudio.load(input_file)

    if rate != sample_rate:
        audio = torchaudio.transforms.Resample(rate, sample_rate)(audio)

    if channels == 1:
        if audio.shape[0] == 2:
            audio = torch.mean(audio, 0)
    else:
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)

    total_length = audio.shape[1]
    window = hop * (dur * 1 - 1)
    stride = window // 2
    rest = stride - (total_length - window)%stride
    audio = torch.cat([audio, torch.zeros((channels, rest))], -1)
    start = 0
    num = np.zeros((channels, audio.shape[1]))
    audio_sum = np.zeros((channels, audio.shape[1]))

    while start < audio.shape[1] - window + 1:
        audio_split = audio[:, start:start + window]
        num[:, start:start + window] += 1

        audio_hat = transform(audio_split, Model, fft, hop, device)

        audio_sum[..., start:start + window] = audio_hat / num[..., start:start + window] + audio_sum[
                                                                                                        ...,
                                                                                                        start:start + window] * (
                                                                num[..., start:start + window] - 1) / num[...,
                                                                                                            start:start + window]
        start += stride

    audio_sum = audio_sum[:,:-rest]

    return audio_sum.T, params


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Music Separation')

    parser.add_argument('input', type=str, nargs='+')

    parser.add_argument('--target', type=str, default='vocals')

    parser.add_argument('--model', type=str, default='./models/vocalacc')

    parser.add_argument('--no-cuda', action='store_true', default=False)

    args, _ = parser.parse_known_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    for input_file in args.input:
        estimate, params = separate(
            input_file,
            target=args.target,
            model_name=args.model,
            device=device,
        )

        output_path = Path('temp', Path(input_file).stem + '_' + Path(args.model).stem)
        output_path.mkdir(exist_ok=True, parents=True)

        sf.write(
            str(output_path / Path(args.target).with_suffix('.wav')),
            estimate,
            params['sample_rate']
        )

