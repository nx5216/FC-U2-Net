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


def eval_main2(args):

    reference_dir = Path(args.root, 'test')
    estimates_dir = Path(args.output_dir, Path(args.model).name, 'estimates')
    output_dir = Path(args.output_dir, Path(args.model).name, 'museval')

    result = pd.DataFrame(columns=['track', 'SDR', 'ISR', 'SIR', 'SAR'])
    result2 = pd.DataFrame(columns=['track', 'SDR', 'ISR', 'SIR', 'SAR'])
    result3 = pd.DataFrame(columns=['track', 'SDR', 'ISR', 'SIR', 'SAR'])
    result4 = pd.DataFrame(columns=['track', 'SDR', 'ISR', 'SIR', 'SAR'])

    estdirs = Path(estimates_dir).iterdir()
    for estdir in tqdm.tqdm(list(estdirs)):
        refdir = Path(reference_dir, estdir.name)
        if refdir.exists():

            ref1, sr = torchaudio.load(str(Path(refdir, 'vocals.wav')))
            est1, sr = torchaudio.load(str(Path(estdir, 'vocals.wav')))
            ref2, sr = torchaudio.load(str(Path(refdir, 'drums.wav')))
            est2, sr = torchaudio.load(str(Path(estdir, 'drums.wav')))
            ref3, sr = torchaudio.load(str(Path(refdir, 'bass.wav')))
            est3, sr = torchaudio.load(str(Path(estdir, 'bass.wav')))
            ref4, sr = torchaudio.load(str(Path(refdir, 'other.wav')))
            est4, sr = torchaudio.load(str(Path(estdir, 'other.wav')))
            # ref2, sr = torchaudio.load(str(Path(refdir, 'accompaniment.wav')))
            # est2, sr = torchaudio.load(str(Path(estdir, 'accompaniment.wav')))
            # ref1, sr = sf.read(str(Path(refdir, 'vocals.wav')), always_2d=True)
            # est1, sr = sf.read(str(Path(estdir, 'vocals.wav')), always_2d=True)
            # ref2, sr = sf.read(str(Path(refdir, 'accompaniment.wav')), always_2d=True)
            # est2, sr = sf.read(str(Path(estdir, 'accompaniment.wav')), always_2d=True)
            mix = ref1 + ref2 + ref3 + ref4
            # mix, sr = torchaudio.load(str(Path(refdir, 'mixture.wav')))

            X = torch.stft(mix, 1024, 512,
                           window=torch.hann_window(1024)).numpy()
            X = X[..., 0] + X[..., 1]*1j
            X = X.transpose(2, 1, 0)

            V1 = torch.norm(torch.stft(est1, 1024, 512,
                           window=torch.hann_window(1024)), dim=-1).numpy()
            V2 = torch.norm(torch.stft(est2, 1024, 512,
                           window=torch.hann_window(1024)), dim=-1).numpy()
            V3 = torch.norm(torch.stft(est3, 1024, 512,
                           window=torch.hann_window(1024)), dim=-1).numpy()
            V4 = torch.norm(torch.stft(est4, 1024, 512,
                           window=torch.hann_window(1024)), dim=-1).numpy()
            V = np.array([V1, V2, V3, V4]).transpose(3, 2, 1, 0)

            Y = norbert.wiener(V, X.astype(np.complex128),
                               1, use_softmask=False)

            est1 = istft(Y[..., 0].T).T
            est2 = istft(Y[..., 1].T).T
            est3 = istft(Y[..., 2].T).T
            est4 = istft(Y[..., 3].T).T

            ref = np.array([ref1.numpy().T, ref2.numpy().T, ref3.numpy().T, ref4.numpy().T])
            est = np.array([est1, est2, est3, est4])

            # ref = np.array([ref1, ref2])
            # est = np.array([est1, est2])
            # est = np.array([est1, est2])

            SDR, ISR, SIR, SAR = museval.evaluate(ref, est, win=sr, hop=sr)
            values = {
                'track': estdir.name,
                "SDR": median_nan(SDR[0]),
                "ISR": median_nan(ISR[0]),
                "SIR": median_nan(SIR[0]),
                "SAR": median_nan(SAR[0])
            }
            result.loc[result.shape[0]] = values

            values2 = {
                'track': estdir.name,
                "SDR": median_nan(SDR[1]),
                "ISR": median_nan(ISR[1]),
                "SIR": median_nan(SIR[1]),
                "SAR": median_nan(SAR[1])
            }
            result2.loc[result2.shape[0]] = values2

            values3 = {
                'track': estdir.name,
                "SDR": median_nan(SDR[2]),
                "ISR": median_nan(ISR[2]),
                "SIR": median_nan(SIR[2]),
                "SAR": median_nan(SAR[2])
            }
            result3.loc[result3.shape[0]] = values3

            values4 = {
                'track': estdir.name,
                "SDR": median_nan(SDR[3]),
                "ISR": median_nan(ISR[3]),
                "SIR": median_nan(SIR[3]),
                "SAR": median_nan(SAR[3])
            }
            result4.loc[result4.shape[0]] = values4
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
    result.to_csv(str(output_dir)+'1.csv',index=0)

    values2 = {
        'track': 'sum',
        "SDR": result2['SDR'].median(),
        "ISR": result2['ISR'].median(),
        "SIR": result2['SIR'].median(),
        "SAR": result2['SAR'].median()
    }
    result2.loc[result2.shape[0]] = values2
    print(list((result2.loc[result2.shape[0] - 1])[1:]))
    result2.to_csv(str(output_dir)+'2.csv',index=0)

    values3 = {
        'track': 'sum',
        "SDR": result3['SDR'].median(),
        "ISR": result3['ISR'].median(),
        "SIR": result3['SIR'].median(),
        "SAR": result3['SAR'].median()
    }
    result3.loc[result3.shape[0]] = values3
    print(list((result3.loc[result3.shape[0] - 1])[1:]))
    result3.to_csv(str(output_dir)+'3.csv',index=0)

    values4 = {
        'track': 'sum',
        "SDR": result4['SDR'].median(),
        "ISR": result4['ISR'].median(),
        "SIR": result4['SIR'].median(),
        "SAR": result4['SAR'].median()
    }
    result4.loc[result4.shape[0]] = values4
    print(list((result4.loc[result4.shape[0] - 1])[1:]))
    result4.to_csv(str(output_dir)+'4.csv',index=0)


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

    # test_main(args)
    # args.target = 'accompaniment'
    # test_main(args)
    # print('ok')
    # eval_main(args)
    # eval_mir1k(args)
    # eval_main2(args)
