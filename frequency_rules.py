class FrequencyRules:
    def __init__(self):
        self.tile_frequencies = []
        self.total = 0
    
    def _increase_size(self, to:int):
        if len(self.tile_frequencies) < to:
            new_size = to - len(self.tile_frequencies)
            self.tile_frequencies.extend([0 for _ in range(new_size)])

    def add(self, tile_ind: int):
        self._increase_size(tile_ind+1)
        self.tile_frequencies[tile_ind] += 1
        self.total += 1

    def get(self, ind:int) -> int:
        return self.tile_frequencies[ind]