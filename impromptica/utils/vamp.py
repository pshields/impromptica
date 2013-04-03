"""Helper functions for working with Vamp plugins."""
import subprocess

import rdflib

from impromptica import settings


def get_input_features(input_filename):
    """Returns a dictionary of the input features."""
    data = {}
    # Run Sonic Annotator on the input audio file.
    # Allocate a temporary file to hold the results.
    # Read in the turtle-formatted results.
    results_filename = input_filename + ".n3"
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

    The segments are returned as indices into the sample audio.
    """
    # TODO Return actual segment labels as well
    results = []
    for segments in g.subjects(rdflib.RDF.type, ns.af['StructuralSegment']):
        for segment in g.objects(segments, ns.event['time']):
            for segment_time in g.objects(segment, ns.tl['at']):
                results.append(float(segment_time[2:len(segment_time) - 1]))

    results = sorted([int(t * settings.SAMPLE_RATE) for t in results])
    return results


def _get_notes(g, ns):
    """Returns the notes from an rdflib graph."""
    # TODO Implement this
    pass
