from typing import List
from src.direction import Direction

class AdjacencyRules:
    def __init__(self):
        self.all_tiles = set()
        self.rules = []
        for _ in Direction:
            self.rules.append({})
    
    def add(self, tile1_ind: int, tile2_ind: int, direction: Direction):
        self.all_tiles.update([tile1_ind, tile2_ind])
        if tile1_ind not in self.rules[direction.value]:
            self.rules[direction.value][tile1_ind] = set()
        self.rules[direction.value][tile1_ind].add(tile2_ind)
    
    def get_compatible(self, tile_ind: int, direction: Direction) -> List[int]:
        dirs = self.rules[direction.value]
        if tile_ind in dirs:
            return dirs[tile_ind]
        else:
            return []

    def get_not_compatible(self, tile_ind:int, direction:Direction) -> List[int]:
        dirs = self.rules[direction.value]
        if tile_ind in dirs:
            return self.all_tiles - dirs[tile_ind]
        else:
            return self.all_tiles