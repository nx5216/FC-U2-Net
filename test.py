import argparse
from pathlib import Path
import torch
import torchaudio
import tqdm
import separate
import soundfile as sf
import time
import museval
import numpy as np
import pandas as pd
import librosa
import norbert
from mir_eval.separation import bss_eval_sources
import scipy.signal
import warnings
warnings.filterwarnings("ignore")

def istft(X, rate=16000, n_fft=1024, n_hopsize=512):
    t, audio = scipy.signal.istft(
        X / (n_fft / 2),
        rate,
        nperseg=n_fft,
        noverlap=n_fft - n_hopsize,
        boundary=True
    )
    return audio


def median_nan(a):
    return np.median(a[~np.isnan(a)])

def test_eval(args):
    tracks = []
    p = Path(args.root, 'test')
    for track_path in tqdm.tqdm(p.iterdir(), disable=True):
        tracks.append(track_path)
    print("files_len", len(tracks))

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    result = pd.DataFrame(columns=['track','SDR','ISR', 'SIR', 'SAR'])
    reference_dir = Path(args.root, 'test')
    output_dir = Path(args.output_dir, Path(args.model).name, 'museval')
    
    for track in tqdm.tqdm(tracks):
        input_file = str(Path(track, args.input))

        estimate, params = separate.separate(
            input_file,
            target=args.target,
            model_name=args.model,
            device=device,
        )

        output_path = Path(args.output_dir, Path(args.model).stem, 'estimates', Path(input_file).parent.name)
        output_path.mkdir(exist_ok=True, parents=True)

        sf.write(
            str(output_path) + '/' + args.target + '.wav',
            estimate,
            params['sample_rate']
        )

        estdir = output_path
        refdir = Path(reference_dir, estdir.name)
        if refdir.exists():
            ref, sr = sf.read(str(Path(refdir, args.target + '.wav')), always_2d=True)
            est, sr = sf.read(str(Path(estdir, args.target + '.wav')), always_2d=True)
            ref = ref[None, ...]
            est = est[None, ...]

            SDR, ISR, SIR, SAR = museval.evaluate(ref, est, win=sr, hop=sr)
            values = {
                'track':estdir.name,
                "SDR": median_nan(SDR[0]),
                "ISR": median_nan(ISR[0]),
                "SIR": median_nan(SIR[0]),
                "SAR": median_nan(SAR[0])
            }
            result.loc[result.shape[0]] = values
        # print(values)
        # break
    values = {
        'track':'sum',
        "SDR": result['SDR'].median(),
        "ISR": result['ISR'].median(),
        "SIR": result['SIR'].median(),
        "SAR": result['SAR'].median()
    }
    result.loc[result.shape[0]] = values
    print(list((result.loc[result.shape[0] - 1])[1:]))
    result.to_csv(str(output_dir)+'.csv',index=0)



def test_main(args):
    tracks = []
    p = Path(args.root, 'test')
    for track_path in tqdm.tqdm(p.iterdir(), disable=True):
        tracks.append(track_path)
    print("files_len", len(tracks))

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print("Using GPU:", use_cuda)
    device = torch.device("cuda" if use_cuda else "cpu")

    start_time = time.time()

    for track in tqdm.tqdm(tracks):
        input_file = str(Path(track, args.input))

        estimate, params = separate.separate(
            input_file,
            target=args.target,
            model_name=args.model,
            device=device,
        )

        output_path = Path(args.output_dir, Path(
            args.model).stem, 'estimates', Path(input_file).parent.name)
        output_path.mkdir(exist_ok=True, parents=True)

        sf.write(
            str(output_path) + '/' + args.target + '.wav',
            estimate,
            params['sample_rate']
        )
        break

    # print(time.time() - start_time)


def eval_main(args):

    reference_dir = Path(args.root, 'test')
    estimates_dir = Path(args.output_dir, Path(args.model).name, 'estimates')
    output_dir = Path(args.output_dir, Path(args.model).name, args.target)

    result = pd.DataFrame(columns=['track', 'SDR', 'ISR', 'SIR', 'SAR'])

    estdirs = Path(estimates_dir).iterdir()
    for estdir in tqdm.tqdm(list(estdirs)):
        refdir = Path(reference_dir, estdir.name)
        if refdir.exists():

            ref, sr = sf.read(
                str(Path(refdir, args.target + '.wav')), always_2d=True)
            est, sr = sf.read(
                str(Path(estdir, args.target + '.wav')), always_2d=True)

            ref = ref[None, ...]
            est = est[None, ...]

            SDR, ISR, SIR, SAR = museval.evaluate(ref, est, win=sr, hop=sr)
            values = {
                'track': estdir.name,
                "SDR": median_nan(SDR[0]),
                "ISR": median_nan(ISR[0]),
                "SIR": median_nan(SIR[0]),
                "SAR": median_nan(SAR[0])
            }
            result.loc[result.shape[0]] = values
            # print(values)
        # break
    values = {
        'track': 'sum',
        "SDR": result['SDR'].median(),
        "ISR": result['ISR'].median(),
        "SIR": result['SIR'].median(),
        "SAR": result['SAR'].median()
    }
    result.loc[result.shape[0]] = values
    print(list((result.loc[result.shape[0] - 1])[1:]))
    result.to_csv(str(output_dir)+'.csv', index=0)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MUSIC test')

    # vocals   accompaniment
    parser.add_argument('--target', type=str, default='vocals')

    parser.add_argument('--model', type=str, default='./models/musdb44_ibm')

    parser.add_argument('--root', type=str, default='J:/musdb16')

    parser.add_argument('--input', type=str, default='mixture.wav')

    parser.add_argument('--output_dir', type=str, default='./eval')

    parser.add_argument('--no-cuda', action='store_true', default=False)

    args, _ = parser.parse_known_args()

    test_eval(args)

