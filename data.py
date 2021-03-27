from utils import load_audio, load_info
from pathlib import Path
import torch.utils.data
import argparse
import random
import musdb
import torch
import tqdm
import glob
import numpy as np


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, audio):
        for t in self.transforms:
            audio = t(audio)
        return audio


def _augment_gain(audio, low=0.25, high=1.25):

    g = low + torch.rand(1) * (high - low)
    return audio * g


def _augment_channelswap(audio):
    """Swap channels of stereo signals with a probability of p=0.5"""
    if audio.shape[0] == 2 and torch.FloatTensor(1).uniform_() < 0.5:
        return torch.flip(audio, [0])
    else:
        return audio


def load_datasets(args):

    source_augmentations = Compose(
        [globals()['_augment_' + aug] for aug in ['gain', 'channelswap']]
    )


    train_dataset = FixedSourcesTrackFolderDataset(
        root = args.root,
        target_file='vocals.wav',
        interferer_files=['drums.wav','bass.wav','other.wav'] ,
        sample_rate=args.sample_rate,
        split='train',
        samples_per_track=args.samples_per_track,
        source_augmentations=source_augmentations,
        seq_duration = (args.dur-1) * args.hop,
        seed = args.seed,
    )
    
    return train_dataset

class FixedSourcesTrackFolderDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            root,
            split='train',
            target_file='vocals.wav',
            interferer_files=['accompaniment.wav'],
            seq_duration=None,
            samples_per_track=16,
            random_chunks=True,
            random_track_mix=True,
            source_augmentations=lambda audio: audio,
            sample_rate=16000,
            seed=42,
    ):
        random.seed(seed)
        self.root = Path(root).expanduser()
        self.split = split
        self.sample_rate = sample_rate
        self.seq_duration = seq_duration
        self.random_track_mix = random_track_mix
        self.random_chunks = random_chunks
        self.source_augmentations = source_augmentations
        # set the input and output files (accept glob)
        self.target_file = target_file
        self.interferer_files = interferer_files
        self.source_files = [self.target_file] + self.interferer_files
        self.tracks = list(self.get_tracks())
        self.samples_per_track = samples_per_track

    def __getitem__(self, index):
        # first, get target track
        index = index // self.samples_per_track

        track_path = self.tracks[index]['path']
        min_duration = self.tracks[index]['min_duration']
        if self.random_chunks:
            start = random.randint(0, min_duration - self.seq_duration)
        else:
            start = 0

        audio_sources = []

        target_audio = load_audio(
            track_path / self.target_file, start=start, dur=self.seq_duration
        )
        target_audio = self.source_augmentations(target_audio)
        audio_sources.append(target_audio)

        for source in self.interferer_files:
            if self.random_track_mix:
                random_idx = random.choice(range(len(self.tracks)))
                track_path = self.tracks[random_idx]['path']
                if self.random_chunks:
                    min_duration = self.tracks[random_idx]['min_duration']
                    start = random.randint(0, min_duration - self.seq_duration)

            audio = load_audio(
                track_path / source, start=start, dur=self.seq_duration
            )

            audio = self.source_augmentations(audio)
            audio_sources.append(audio)

        stems = torch.stack(audio_sources)
        # # apply linear mix over source index=0
        x = stems.sum(0)

        # y = stems.reshape(-1, stems.size(2))
        y = stems[0]

        return x, y

    def __len__(self):
        return len(self.tracks) * self.samples_per_track

    def get_tracks(self):
        p = Path(self.root, self.split)
        for track_path in tqdm.tqdm(p.iterdir(), disable=True):
            if track_path.is_dir():
                source_paths = [track_path / s for s in self.source_files]
                if not all(sp.exists() for sp in source_paths):
                    print("exclude track ", track_path)
                    continue

                if self.seq_duration is not None:
                    infos = list(map(load_info, source_paths))
                    # get minimum duration of track
                    min_duration = min(i['duration'] for i in infos)
                    if min_duration > self.seq_duration:
                        yield ({
                            'path': track_path,
                            'min_duration': min_duration
                        })
                else:
                    yield ({'path': track_path, 'min_duration': None})



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataset Test')
    parser.add_argument('--root', type=str, default='dataset')

    parser.add_argument('--target', type=str, default='vocals')

    parser.add_argument('--dur', type=int, default=256)

    parser.add_argument('--fft', type=int, default=1024)
    parser.add_argument('--hop', type=int, default=512)
    parser.add_argument('--seq-dur', type=float, default=6.0)
    parser.add_argument('--batch-size', type=int, default=16)

    parser.add_argument('--sample-rate', type=int, default=16000)

    parser.add_argument('--samples-per-track', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42, metavar='S')
    args, _ = parser.parse_known_args()

    train_dataset,args = load_datasets(parser, args)

    print("Number of train samples: ", len(train_dataset))

    # iterate over dataloader

    train_sampler = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2,
    )

    for x, y in tqdm.tqdm(train_sampler):
        pass
