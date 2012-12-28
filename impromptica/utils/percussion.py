"""Utilities for working with percussion samples."""
import os
from xml.etree import ElementTree as ET

import numpy
from scikits.audiolab import Sndfile, Format

from impromptica import settings
from impromptica.utils import sound


_DEFAULT_PERCUSSION = {
    0: [0, 0.5],
    3: [0.25, 0.75],
}


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


def render_percussion(samples, samples_per_second, beats_per_minute, sounds):
    """Renders percussion onto the given samples.

    This function assumes samples[0] is the start of a new beat.

    This function currently uses a predefined drum loop for the generated
    purcussion. In the future it should use more advanced logic to generate
    percussion appropriate to the piece.
    """
    # Calculate some quantities we'll want to use.
    n = len(samples)
    beats_per_second = beats_per_minute / 60.0
    # Assume beats per measure.
    beats_per_second /= 4.
    samples_per_beat = int(samples_per_second / beats_per_second)
    # `number_of_beats` is the maximum number of beats that we need to render.
    # The last beat might not need to be rendered all of the way.
    number_of_beats = n / samples_per_beat
    # Render the percussion onto a blank track.
    percussion_samples = numpy.zeros(len(samples), dtype=numpy.double)
    for beat in range(number_of_beats):
        for instrument, inter_beat_onsets in _DEFAULT_PERCUSSION.iteritems():
            sound_samples = sounds[instrument]
            for inter_beat_onset in inter_beat_onsets:
                # Calculate the onset for the samples for this sound.
                onset = int((float(beat) + inter_beat_onset)
                            * samples_per_beat)
                if onset + len(sound_samples) > n:
                    sound_samples = sound_samples[:n - onset]
                # Create samples for this sound at the appropriate offset.
                percussion_samples[onset:onset + len(sound_samples)
                                   ] += sound_samples
    # Merge the percussion and original audio.
    sound.merge_audio(samples, percussion_samples)
