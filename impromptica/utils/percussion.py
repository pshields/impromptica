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


def render_percussion(samples, tatums, tactus, measures, sounds):
    """Renders percussion onto the given samples.

    This function assumes samples[0] is the start of a new beat.

    This function currently uses a predefined drum loop for the generated
    purcussion. In the future it should use more advanced logic to generate
    percussion appropriate to the piece.
    """
    # Calculate some quantities we'll want to use.
    n = len(samples)
    # Render the percussion onto a blank track.
    percussion_samples = np.zeros(n, dtype=np.double)
    sound_samples = sounds[15]
    for onset in measures:
        # Calculate the onset for the samples for this sound.
        if onset + len(sound_samples) > n:
            sound_samples = sound_samples[:n - onset]
        # Create samples for this sound at the appropriate offset.
        percussion_samples[onset:onset + len(sound_samples)] += sound_samples
    sound_samples = sounds[11]
    for onset in tactus:
        # Calculate the onset for the samples for this sound.
        if onset + len(sound_samples) > n:
            sound_samples = sound_samples[:n - onset]
        # Create samples for this sound at the appropriate offset.
        percussion_samples[onset:onset + len(sound_samples)] += sound_samples
    # Wait for tatum detection to get a little bit better before adding this.
    # sound_samples = sounds[4]
    #for onset in tatums:
    #    # Calculate the onset for the samples for this sound.
    #    if onset + len(sound_samples) > n:
    #        sound_samples = sound_samples[:n - onset]
        # Create samples for this sound at the appropriate offset.
    #    percussion_samples[onset:onset + len(sound_samples)] += sound_samples

    # Merge the percussion and original audio.
    percussion_samples /= np.max(percussion_samples)
    sound.merge_audio(samples, percussion_samples)
