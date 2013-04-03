"""Helper functions for working with Vamp plugins."""
import subprocess

import rdflib

from impromptica import settings


def get_beats(input_filename):
    """Returns the indices of the beats of the piece."""
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
    vamp = rdflib.Namespace('http://purl.org/ontology/vamp/')
    af = rdflib.Namespace('http://purl.org/ontology/af/')
    event = rdflib.Namespace('http://purl.org/NET/c4dm/event.owl#')
    tl = rdflib.Namespace('http://purl.org/NET/c4dm/timeline.owl#')

    g = rdflib.Graph()
    g.parse(results_filename, format='turtle')

    results = []
    for beats in g.subjects(rdflib.RDF.type, af['Beat']):
        for moment in g.objects(beats, event['time']):
            for moment_time in g.objects(moment, tl['at']):
                results.append(float(moment_time[2:len(moment_time) - 1]))

    results = [int(t * settings.SAMPLE_RATE) for t in results]

    return results
