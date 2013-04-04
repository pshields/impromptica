"""Helper functions for working with Vamp plugins."""
import subprocess

import rdflib

from impromptica import settings


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
    data['notes'] = _get_notes(g, NS)
    data['onsets'] = _get_onsets(g, NS)
    data['segments'] = _get_segments(g, NS)
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


def _get_segments(g, ns):
    """Returns the segments from an rdflib graph.

    The segments are returned as (index, label) pairs.
    """
    results = []
    for segment in g.subjects(rdflib.RDF.type, ns.af['StructuralSegment']):
        start = g.objects(segment, ns.event['time']).next()
        segment_time = g.objects(start, ns.tl['at']).next()
        segment_time = int(float(segment_time[2:len(segment_time) - 1]) * (
            settings.SAMPLE_RATE))
        segment_label = int(g.value(segment, ns.af['feature']).toPython())
        results.append((segment_time, segment_label))

    results = sorted(results, key=lambda s: s[0])
    return results


def _get_notes(g, ns):
    """Returns the notes from an rdflib graph.
    
    Notes are (note_midi_value, onset_index, duration_in_samples) triples.
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
