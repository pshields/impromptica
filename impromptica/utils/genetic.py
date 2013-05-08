"""Genetic algorithm implementation for music accompaniment."""
import random

from impromptica import settings
from impromptica.utils import generation
from impromptica.utils import notes
from impromptica.utils import sound
from impromptica import probdata

class Individual(object):
    """A possible accompaniment."""

    def __init__(self, original_notes, notes, percussion):
        """Initializes an `Individual` from existing features.
        This is used to create the initial individuals, which are copies of
        the detected features of the input audio.
        """
        self.orig_notes = original_notes
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
        """Performs a mutation in-place on this ."""
        # TODO Implement this in better detail. Use multiple mutations.
        #print 'Mutating'
        x = random.random()
        if x <= 0.2:
            self.mutate_remove_note()
        elif x <= 0.4:
            self.mutate_add_note()
        elif x <= 0.6:
            self.mutate_add_fifth()
        elif x <= 0.8:
            self.mutate_add_major_third()
        else:
            self.mutate_add_minor_third()

    def mutate_remove_note(self):
        """Randomly remove a note from this individual, in-place."""
        # Calculate the number of notes in this individual.
        #print 'Removing note'
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
       # print 'Note removed'
    
    def mutate_add_note(self):
        """Add a note in the same key to a random tatum"""
        #print 'Adding Note'
        scale = sound.get_key(self.notes)
        tactus_index = random.randint(1, (len(self.notes)-1)/4)
        tatum_index = tactus_index*4 - 1
        if len(self.notes[tatum_index])>1:
            return
        note_index = random.randint(0, (len(scale)-1))
        note = scale[note_index]
        if len(self.notes[tatum_index])==0:
            duration = 1
        else:
            duration = self.notes[tatum_index][0].duration
        self.notes[tatum_index].append(notes.Note(0,48+note,duration))
        #print 'Note Added'
        
    def mutate_add_fifth(self):
        """Add a fifth note in the same key to a random tatum"""
        #print 'Adding Fifth'
        k = []
        while len(k)==0:
            tatum_index =  random.randint(1, (len(self.notes)-1))
            k = self.notes[tatum_index]
        if len(self.notes[tatum_index])>1:
            return
        index_in_tatum = random.randint(0,len(k)-1)
        d = sound.make_scales()
        note = sound.midival_note(self.notes[tatum_index][index_in_tatum].midi_note)[0]
        fifth = d[str(note)][4]
        duration = self.notes[tatum_index][0].duration
        self.notes[tatum_index].append(notes.Note(0,72+fifth,duration))
        #print 'Fifth Added'
        
    def mutate_add_major_third(self):
        """Add a major third note in the same key to a random tatum"""
       # print 'Adding Thrid'
        k = []
        while len(k)==0:
            tatum_index =  random.randint(1, (len(self.notes)-1))
            k = self.notes[tatum_index]
        if len(self.notes[tatum_index])>1:
            return
        index_in_tatum = random.randint(0,len(k)-1)
        note = sound.midival_note(self.notes[tatum_index][index_in_tatum].midi_note)[0]
        third = (note+4)%12
        duration = self.notes[tatum_index][0].duration
        self.notes[tatum_index].append(notes.Note(0,72+third,duration))
        #print 'Third Added'
        
    def mutate_add_minor_third(self):
       # print 'Adding minor third'
        """Add a minor note in the same key to a random tatum"""
        k = []
        while len(k)==0:
            tatum_index =  random.randint(1, (len(self.notes)-1))
            k = self.notes[tatum_index]
        if len(self.notes[tatum_index])>1:
            return
        index_in_tatum = random.randint(0,len(k)-1)
        note = sound.midival_note(self.notes[tatum_index][index_in_tatum].midi_note)[0]
        third = (note+3)%12
        duration = self.notes[tatum_index][0].duration
        self.notes[tatum_index].append(notes.Note(0,72+third,duration))
        #print 'Minor Third Added'
    
    def get_fitness (self):
        notes = self.notes
        original = self.orig_notes
        notes_on_key = 0
        note_density = 0
        num_notes = 0
        non_rest_tatums = 0
        markov_score = 0
        prob = []
        key = sound.get_key(self.notes)
        key_num = key[0]
        isMinor = False
        if key[2] == key[0]+3:
            isMinor = True
            prob = probdata.KP_MINOR_KEY_PROFILE_DATA
        else:
            prob = probdata.KP_MAJOR_KEY_PROFILE_DATA
            
        prev = 0
        for index, i in enumerate(self.notes):
            if len(i)!=0:
                note_val = sound.midival_note(i[0].midi_note)[0]
                non_rest_tatums += 1
                num_notes = num_notes+len(i)
                for note in i:
                    if sound.midival_note(note.midi_note)[0] in key:
                        notes_on_key += 1
                if index == 0:
                    prev = note_val
                    continue
                prob_ind = relative_distance (prev, note_val)
                #print prob_ind
                markov_score = markov_score+prob[prob_ind]*1000
                prev = note_val
            
                
        markov_score = float(markov_score)/float(non_rest_tatums)
        note_density = float(num_notes)/float(non_rest_tatums)
        dense_penalty = note_density - 1.0
        off_key_penalty = (1.0 - float(notes_on_key)/float(num_notes))*100
        print markov_score/float(non_rest_tatums), dense_penalty, off_key_penalty
        fitness = markov_score/float(non_rest_tatums)-dense_penalty-off_key_penalty
        return fitness
        
class Population(object):
    """A population of possible accompaniments.

    We use genetic algorithms to evolve the population through a series of
    generations. At the end, the best individual in the last generation of the
    population is returned as the candidate accompaniment to be rendered onto
    the input audio file.
    """

    def __init__(self, original_seed, seed, start_randomly=False, size=settings.POPULATION_SIZE):
        """ To initialize a population we first take the seed i.e. the 
        input notes in the form of a tatum array. Then we  them randomly
        to form our first population
        """
        self.size = size
        self.individuals = []
        # Initializing our population
        if start_randomly:
            for i in range(self.size):
                self.individuals.append(Individual.rand(len(seed)))
        else:
            for i in range(self.size):
                ind = Individual(original_seed, seed, [])
                ind.mutate()
                self.individuals.append(ind)

    def evolve(self, rounds=settings.ROUNDS_OF_EVOLUTION):
        for i in range(rounds):
            print 'Evolution round:', i
            for j in range(self.size):
                self.individuals[0].mutate()

            if i % 128 == 0:
                print("%3f%% complete" % (100. * (float(i) / float(rounds))))

    def most_fit(self):
        """Returns the most-fit member of the population."""
        # TODO
        max = self.individuals[0].get_fitness()
        best = self.individuals[0]
        for i in self.individuals:
            fit = i.get_fitness() 
            if  fit > max:
                max = fit
                best = i
        return best
    
def relative_distance (key_num, note):
    #print key_num, note
    if note >= key_num:
        return note - key_num
    else:
        return note + 12-key_num
        
def change_to_monophonic(notes):
    result = []
    for i in notes:
            if len(i)!=0:
                index = random.randint (0, len(i)-1)
                result.append([i[index]])
            else:
                result.append([])
    return result

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
        p = Population(grid, change_to_monophonic(grid))
        p.evolve()
        results.append(p.most_fit())
        print results
        print p.most_fit()
    return results

