"""Genetic algorithm implementation for music accompaniment."""


def evolve(input_features, population_size):
    """Given the features of an input audio file, returns accompaniment."""
    # Seed the initial population via mutation over the input features.
    initial_accompaniment = {}

    # Populate the 'notes' parameter of the first accompaniment individual.
    # We use a representation of notes as a list of lists of note attacks at
    # each tatum.
    beat_number = 0
    tatums_per_beat = 4

    initial_accompaniment['notes'] = [[] for i in range(len(input_features['beats']) * (
        tatums_per_beat))]
    for midi_note, start_index, duration in input_features['notes']:
        print(midi_note)
        while beat_number < len(input_features['beats']) and (
            input_features['beats'][beat_number] < start_index):
            beat_number += 1
        if beat_number >= len(input_features['beats']):
            break
        samples_this_beat = input_features['beats'][beat_number + 1] - (
            input_features['beats'][beat_number])
        note_tatum_length = max(1, (duration * tatums_per_beat) / samples_this_beat)
        offset = int((start_index - (input_features['beats'][beat_number])) / (
            samples_this_beat / tatums_per_beat))
        initial_accompaniment['notes'][
            beat_number * tatums_per_beat + offset].append(
                [midi_note, note_tatum_length])

    population = [initial_accompaniment]
    print(population[0])
    return population[0]
