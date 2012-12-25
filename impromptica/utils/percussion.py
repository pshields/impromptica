"""Utilities for working with percussion samples."""
import os
from xml.etree import ElementTree as ET

from scikits.audiolab import Sndfile


def get_drumkit_samples(drumkit_dir):
    """Returns a list of drumkit sound samples arrays."""
    settings_filename = os.path.join(drumkit_dir, 'drumkit.xml')
    drumkit = ET.parse(settings_filename)
    result = []
    for node in drumkit.iter('instrument'):
        f = Sndfile(os.path.join(drumkit_dir, node.find('filename').text), 'r')
        samples = f.read_frames(f.nframes)
        result.append(samples)
    return result
