"""Genetic algorithm implementation for music accompaniment."""
import random

from impromptica import settings
from impromptica.utils import generation
from impromptica.utils import notes


class Individual(object):
    """A possible accompaniment."""

    def __init__(self, notes, percussion):
        """Initializes an `Individual` from existing features.

        This is used to create the initial individuals, which are copies of
        the detected features of the input audio.
        """
        self.notes = notes
        self.percussion = percussion

    @classmethod
    def rand(cls, num_tatums):
        """Generates a random individual."""
        obj = cls([], [])
        obj.notes = [[] for i in range(num_tatums)]
        n = n0 = notes.Note(0, random.randint(40, 80), 1)
        obj.notes[0].append(n0)
        for i in range(1, num_tatums):
            n = notes.Note(
                0, generation.generate_note(
                    n.midi_note, n0.midi_note, (n0.midi_note, 1)), 1)
            obj.notes[i].append(n)
        return obj

    def mutate(self):
        """Performs a mutation in-place on this individual."""
        # TODO Implement this in better detail. Use multiple mutations.
        x = random.random()
        if x <= 0.01:
            self.mutate_remove_note()

    def mutate_remove_note(self):
        """Randomly remove a note from this individual, in-place."""
        # Calculate the number of notes in this individual.
        num_notes = 0
        for note_set in self.notes:
            for note in note_set:
                num_notes += 1
        # If there aren't any notes in this individual, quit early.
        if num_notes == 0:
            return
        # Select one note randomly for removal.
        n = random.randint(1, num_notes)
        # Remove it.
        for note_set in self.notes:
            for i in range(len(note_set)):
                n -= 1
                if n == 0:
                    note_set.pop(i)
                    break
            if n <= 0:
                break


class Population(object):
    """A population of possible accompaniments.

    We use genetic algorithms to evolve the population through a series of
    generations. At the end, the best individual in the last generation of the
    population is returned as the candidate accompaniment to be rendered onto
    the input audio file.
    """

    def __init__(self, n=settings.POPULATION_SIZE):
        self.n = n

    def seed(self, grid, start_randomly=False):
        """Seeds the population for the given segment.
        
        `grid` is a tatum array of notes.

        `start_randomly` is a boolean representing whether the individuals
        should be created randomly or from the input features.
        """
        self.initial_notes = grid
        self.p = []  # `p` holds the list of individuals
        for i in range(self.n):
            if start_randomly:
                ind = Individual.rand(len(grid))
            else:
                ind = Individual(grid, [])
            self.p.append(ind)

    def evolve(self, rounds=settings.ROUNDS_OF_EVOLUTION):
        for i in range(rounds):
            for j in range(self.n):
                self.p[j].mutate()

            if i % 128 == 0:
                print("%3f%% complete" % (100. * (float(i) / float(rounds))))

    def most_fit(self):
        """Returns the most-fit member of the population."""
        # TODO
        return self.p[0]
        

def get_genetic_accompaniment(input_notes):
    """Given the features of an input audio file, returns accompaniment.

    The general strategy is to create accompaniment for each segment of the
    piece separately, using the longest instance of each segment to get input
    features.
    
    `input_notes` is a list of tatum arrays for the longest instance of each
    segment.

    See the `vamp.py` module in this folder for more information.
    """
    results = []
    # Create a population for each segment.
    for i, grid in enumerate(input_notes):
        print("Accompanying segment %d" % (i))
        p = Population()
        p.seed(grid, start_randomly=True)
        p.evolve()
        results.append(p.most_fit())
    return results
