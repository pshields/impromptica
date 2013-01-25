"""Utilities for working with percussion samples."""
import os
from xml.etree import ElementTree as ET

import numpy as np
from scikits.audiolab import Sndfile

from impromptica import settings
from impromptica.utils import sound


def get_drumkit_samples(drumkit_dir=settings.DRUMKIT_DIR):
    """Returns a list of drumkit sounds.

    Each sound is an array of amplituded samples.
    """
    settings_filename = os.path.join(drumkit_dir, 'drumkit.xml')
    drumkit = ET.parse(settings_filename)
    result = []
    for node in drumkit.iter('instrument'):
        f = Sndfile(os.path.join(drumkit_dir, node.find('filename').text), 'r')
        samples = f.read_frames(f.nframes)
        # Combine the audio into a single track.
        if samples.ndim > 1:
            samples = samples.sum(axis=1)
        result.append(samples)
    return result


def render_metronome(samples, levels, soundbank, play_tatums):
    """Renders percussive sounds at the beats of given metrical levels.

    `levels` is a list of lists of beat indices, where each top-level list
    represents the beats at a particular metrical level.

    `play_tatums` is a boolean indicating whether or not to play sounds at the
    tatum level.
    """
    # Select a crash symbol to play on measure beats, a cowbell to play on
    # tactus beats, and a closed hi-hat to play on tatum beats.
    sounds = [soundbank[15], soundbank[11], soundbank[6]]
    if not play_tatums:
        sounds.pop()
        levels = levels[:-1]
    for s, beats in zip(sounds, levels):
        render_sounds(samples, beats, s)


def render_sounds(samples, indices, sound_samples):
    """Adds sound to samples at the given starting indices."""
    result = np.zeros(samples.shape[0])
    for onset in indices:
        # Crop the sound if necessary.
            if onset + sound_samples.shape[0] > samples.shape[0]:
                sound_samples = sound_samples[:samples.shape[0] - onset]
            # Create samples for this sound at the appropriate offset.
            result[onset:onset + sound_samples.shape[0]] += sound_samples
    # Normalize the amplitude of the resulting track.
    max_amplitude = np.max(result)
    if max_amplitude > 0:
        result /= max_amplitude
    # Merge the samples with the provided input track.
    sound.merge_audio(samples, result)
