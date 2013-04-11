"""Genetic algorithm implementation for music accompaniment."""
from impromptica import settings


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
    def random(cls):
        """Generates a random individual."""
        # TODO
        pass

    def mutate(self):
        """Performs a mutation in-place on this individual."""
        # TODO
        pass


class Population(object):
    """A population of possible accompaniments.

    We use genetic algorithms to evolve the population through a series of
    generations. At the end, the best individual in the last generation of the
    population is returned as the candidate accompaniment to be rendered onto
    the input audio file.
    """

    def __init__(self, n=settings.POPULATION_SIZE):
        self.n = n

    def seed(self, grid):
        """Seeds the population for the given segment.
        
        `grid` is a tatum array of notes.
        """
        self.initial_notes = grid
        self.p = []  # `p` holds the list of individuals
        for i in range(self.n):
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
        p.seed(grid)
        p.evolve()
        results.append(p.most_fit())
    return results
