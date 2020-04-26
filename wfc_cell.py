import numpy as np
import random

class WFCCell:
    def __init__(self, num_tiles, frequencies, adjacencies, adjacency_tile_counts):
        self.num_tiles = num_tiles
        self.frequencies = frequencies
        self.adjacencies = adjacencies
        self.adjacency_tile_counts = adjacency_tile_counts

        self.possible = [True for _ in range(num_tiles)]
        self.possible_ind = [i for i, val in enumerate(self.possible) if val]

        self.collapsed = False

        self.entropy_noise = (random.random()-0.5)*.00000001
        self.sum_possible_tile_weights = self.compute_tile_frequency()
        self.sum_possible_tile_weights_log = self.compute_log_tile_frequency()

        
    def choose_tile_index(self):
        remaining = random.random()*self.sum_possible_tile_weights

        for index in self.possible_ind:
            weight = self.frequencies.get(index)
            if remaining >= weight:
                remaining -= weight
            else:
                return index

    def compute_tile_frequency(self):
        possible_freq_values = [self.frequencies.get(i) for i in self.possible_ind]
        return sum(possible_freq_values)
    
    def compute_log_tile_frequency(self):
        possible_freq_values = np.array([self.frequencies.get(i) for i in self.possible_ind])
        return np.sum(possible_freq_values * np.log2(possible_freq_values))
    
    def entropy(self):
        if self.sum_possible_tile_weights:
            return np.log2(self.sum_possible_tile_weights) - (self.sum_possible_tile_weights_log / self.sum_possible_tile_weights) + self.entropy_noise
        else:
            return self.entropy_noise
        
    def remove_possibility(self, tile_index):
        if not self.possible[tile_index]:
            return

        self.possible[tile_index] = False
        self.possible_ind.remove(tile_index)

        freq = self.frequencies.get(tile_index)
        self.sum_possible_tile_weights -= freq
        self.sum_possible_tile_weights_log -= freq * np.log2(freq)

    def collapse(self):
        tile_index_to_lock_in = self.choose_tile_index()
        
        old_possible_ind = self.possible_ind

        self.possible = [False for _ in range(self.num_tiles)]
        self.possible[tile_index_to_lock_in] = True
        self.possible_ind = [tile_index_to_lock_in]
        self.collapsed = True

        return (tile_index_to_lock_in, old_possible_ind)