"""Helper functions for working with Vamp plugins."""
import subprocess

import rdflib

from impromptica import settings
from impromptica.utils import notes


def get_input_features(input_filename, use_cached_features=False):
    """Returns a dictionary of the input features."""
    data = {}
    # Run Sonic Annotator on the input audio file.
    # Allocate a temporary file to hold the results.
    # Read in the turtle-formatted results.
    results_filename = input_filename + ".n3"
    if not use_cached_features:
        subprocess.check_call(['rm', '-f', results_filename])
        args = ['sonic-annotator', '-T',
                settings.SONIC_ANNOTATOR_SETTINGS_FILENAME, '-w', 'rdf',
                '--rdf-one-file', results_filename, input_filename]
        subprocess.check_call(args)

    # Set up rdflib namespaces.
    class NS:
        vamp = rdflib.Namespace('http://purl.org/ontology/vamp/')
        af = rdflib.Namespace('http://purl.org/ontology/af/')
        event = rdflib.Namespace('http://purl.org/NET/c4dm/event.owl#')
        tl = rdflib.Namespace('http://purl.org/NET/c4dm/timeline.owl#')

    g = rdflib.Graph()
    g.parse(results_filename, format='turtle')

    data['beats'] = _get_beats(g, NS)
    data['segment_onsets'] = _get_segment_onsets(g, NS, data['beats'])
    data['segments'] = get_segment_instances(
        g, NS, data['beats'], data['segment_onsets'])
    data['notes'] = get_notes_in_longest_segment_instances(
        g, NS, data['beats'], data['segments'])
    data['onsets'] = _get_onsets(g, NS)
    return data


def _get_beats(g, ns):
    """Returns the beats from an rdflib graph.

    The beats are returned as indices into the sample audio.
    """
    results = []
    for beats in g.subjects(rdflib.RDF.type, ns.af['Beat']):
        for moment in g.objects(beats, ns.event['time']):
            for moment_time in g.objects(moment, ns.tl['at']):
                results.append(float(moment_time[2:len(moment_time) - 1]))

    results = sorted([int(t * settings.SAMPLE_RATE) for t in results])
    return results


def _get_onsets(g, ns):
    """Returns the onsets from an rdflib graph.

    The onsets are returned as indices into the sample audio.
    """
    results = []
    for onsets in g.subjects(rdflib.RDF.type, ns.af['Onset']):
        for onset in g.objects(onsets, ns.event['time']):
            for onset_time in g.objects(onset, ns.tl['at']):
                results.append(float(onset_time[2:len(onset_time) - 1]))

    results = sorted([int(t * settings.SAMPLE_RATE) for t in results])
    return results


def _get_segment_onsets(g, ns, beats):
    """Returns the segments from an rdflib graph.

    The segments are returned as (index, label) pairs.

    Segment indices are moved to the nearest beat index.
    """
    results = []
    for segment in g.subjects(rdflib.RDF.type, ns.af['StructuralSegment']):
        start = g.objects(segment, ns.event['time']).next()
        segment_time = g.objects(start, ns.tl['at']).next()
        segment_time = int(float(segment_time[2:len(segment_time) - 1]) * (
            settings.SAMPLE_RATE))
        segment_label = int(g.value(segment, ns.af['feature']).toPython())
        results.append([segment_time, segment_label])

    results = sorted(results, key=lambda s: s[0])
    # Move segment indices to the nearest beat index.
    # Find the closest beat to the start index.
    for i in range(len(results)):
        results[i][0] = beats[min(enumerate(beats),
                              key=lambda x: abs(x[1] - results[i][0]))[0]]
    # At this point, the segment labels are not gauranteed to be the
    # consecutive nonnegative integers 0...n - 1. It's convenient for them to have
    # such labels. Rename the labels so that they take on the values 0...n - 1.
    segment_ids = sorted(set(x[1] for x in results))
    new_ids = {}
    for i, el in enumerate(segment_ids):
        new_ids[el] = i
    for s in results:
        s[1] = new_ids[s[1]]
    return results


def get_segment_instances(g, ns, b, s):
    """Returns the longest continuous instance of each segment in the graph.

    The segments are returned as (start, end) pairs of sample indices which
    represent the nearest beat indices of the found segment instances.

    `b` is a list of the sample indices of the beats of the piece.

    `s` is a list of (index, label) segment onset tuples.
    """
    # First, find the longest instance of each segment.
    num_segments = max(x[1] for x in s) + 1
    longest_durations = {}
    longest_segments = {}
    for i in range(len(s)):
        first_beat = s[i][0]
        last_beat = b[-1]
        if i < len(s) - 1:
            last_beat = b[min(enumerate(b),
                              key=lambda x: abs(x[1] - s[i + 1][0]))[0]]
        duration = last_beat - first_beat
        if duration > longest_durations.get(s[i][1], 0):
            longest_durations[s[i][1]] = duration
            longest_segments[s[i][1]] = [first_beat, last_beat]
    return [longest_segments[i] for i in range(num_segments)]


def _get_notes(g, ns):
    """Returns the notes from an rdflib graph.
    
    `ns` is a wrapper of various rdf namespaces.

    The result is a list of (midi_note, start_index, duration_in_samples)
    triples.
    """
    results = []
    for note in g.subjects(rdflib.RDF.type, ns.af['Note']):
        time_details = g.objects(note, ns.event['time']).next()
        start_time = g.objects(time_details, ns.tl['beginsAt']).next()
        duration = g.objects(time_details, ns.tl['duration']).next()
        label = int(g.value(note, ns.af['feature']).toPython())
        start_time = int(float(start_time[2:len(start_time) - 1]) * (
            settings.SAMPLE_RATE))
        duration = int(float(duration[2:len(duration) - 1]) * (
            settings.SAMPLE_RATE))
        results.append((label, start_time, duration))

    results = sorted(results, key=lambda s: s[1])
    return results


def get_notes_in_longest_segment_instances(g, ns, beats, segments):
    """Returns tatum arrays of notes.

    `segments` is a list of (first_index, last_index) tuples representing the
    longest contiguous instance of each segment. `first_index` and
    `last_index` are gauranteed to lie on beats. `segments` is indexed by
    segment id.

    The notes are returned as a list of tatum arrays indexed by segment number.
    See `notes.py` for details on the structure of the tatum arrays.
    """
    xs = _get_notes(g, ns)
    results = []
    for i, seg in enumerate(segments):
        start, end = seg
        num_beats = beats.index(end) - beats.index(start)
        num_tatums = num_beats * settings.DEFAULT_TATUMS_PER_BEAT
        grid = [[] for x in range(num_tatums)]
        for midi_note, start_index, duration in xs:
            # Disregard notes not in the longest instance of this segment.
            if start_index < start or start_index > end:
                continue
            # Find the closest beat to this note's onset.
            beat_number = min(enumerate(beats),
                              key=lambda x: abs(x[1] - start_index))[0]
            # Calculate the duration between this and the preceeding or
            # succeeding beats.
            comes_before = (start_index < beats[beat_number])
            beat_duration = 0
            if comes_before:
                # Disregard notes that come before the very first beat of this
                # segment instance.
                if beats[beat_number] < start:
                    continue
                beat_duration = beats[beat_number] - beats[beat_number - 1]
            else:
                # Disregard notes that come after the very last beat of this
                # segment instance.
                if beats[beat_number] > end:
                    continue
                beat_duration = beats[beat_number + 1] - beats[beat_number]
            # Calculate the offset from the beat as a number of tatums.
            ratio = float(abs(start_index - beats[beat_number])) / float(
                beat_duration)
            tatum_offsets = [int(float(x) / settings.DEFAULT_TATUMS_PER_BEAT)
                             for x in range(settings.DEFAULT_TATUMS_PER_BEAT)]
            offset = min(enumerate(tatum_offsets),
                         key=lambda x: abs(x[1] - ratio))[0]
            if comes_before:
                offset = -offset
            # Calculate the offset between this beat and the first beat of
            # this segment instance.
            beat_offset = beat_number - beats.index(start)
            tatum_offset = beat_offset * settings.DEFAULT_TATUMS_PER_BEAT + (
                offset)
            # If the tatum offset would place this note on the very last beat
            # of this segment instance, discard it.
            if tatum_offset == len(grid):
                continue
            duration_in_tatums = int(float(duration) * (
                settings.DEFAULT_TATUMS_PER_BEAT) / beat_duration)
            n = notes.Note(0, midi_note, duration_in_tatums)
            grid[tatum_offset].append(n)
        results.append(grid)
    return results
