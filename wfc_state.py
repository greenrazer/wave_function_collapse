import random
import heapq
import copy
import numpy as np
from direction import Direction
from wfc_cell import WFCCell

class WFCState:
    def __init__(self, num_tiles, adjacencies, frequencies, output_size):
        self.num_tiles = num_tiles
        self.frequencies = frequencies
        self.adjacencies = adjacencies
        self.output_size = output_size

        self.output_ind_size = np.prod(output_size)

        self.uncollapsed = 0

        self.grid = []
        self.entropy_heap = []
        self.removal_stack = []

        self.init_tile_adjacency_count = []

        self.init_adjacent_tile_counts()
        self.init_grid()

    def init_adjacent_tile_counts(self):
        for tile1_ind in range(self.num_tiles):
            counts = [0,0,0,0]
            for direction in Direction:
                counts[direction.value] = len(self.adjacencies.get_compatible(tile1_ind, direction))
            self.init_tile_adjacency_count.append(counts)
        

    def init_grid(self):
        for i in range(self.output_ind_size):
            cell = WFCCell(self.num_tiles, self.frequencies,self.adjacencies, copy.deepcopy(self.init_tile_adjacency_count))
            if not cell.collapsed:
                self.grid.append(cell)
                heapq.heappush(self.entropy_heap, (cell.entropy(), i))
                self.uncollapsed += 1

    def to_index(self, coord):
        return coord[0] + self.output_size[0]*coord[1]
    
    def to_coord(self, index):
        return [index % self.output_size[0], index // self.output_size[0]]
    
    def choose_next_cell(self) -> int:
        while len(self.entropy_heap):
            (_, cell_index) = heapq.heappop(self.entropy_heap)
            cell = self.grid[cell_index]
            if not cell.collapsed:
                return cell_index
        raise RuntimeError("Entropy heap empty")

    def collapse_cell_at(self, loc):
        (collapsed_to, possible_ind) = self.grid[loc].collapse()
        for index in possible_ind:
            if index != collapsed_to:
                self.removal_stack.append((index,loc))

    def get_neighbor_index(self, index, direction, wrap = True):
        neighboor_coord = self.to_coord(index)

        if direction == Direction.UP:
            neighboor_coord[1] = neighboor_coord[1] + 1
        elif direction == Direction.DOWN:
            neighboor_coord[1] = neighboor_coord[1] - 1
        elif direction == Direction.LEFT:
            neighboor_coord[0] = neighboor_coord[0] - 1
        elif direction == Direction.RIGHT:
            neighboor_coord[0] = neighboor_coord[0] + 1

        if wrap:
            neighboor_coord[0] %= self.output_size[0]
            neighboor_coord[1] %= self.output_size[1]

        return self.to_index(neighboor_coord)

    def propagate(self):
        while self.removal_stack:
            (tile_index, cell_index) = self.removal_stack.pop()
            for direction in Direction:
                neighbor_index = self.get_neighbor_index(cell_index, direction)
                try:
                    neighbor_cell = self.grid[neighbor_index]
                except IndexError:
                    continue

                if neighbor_cell.collapsed:
                    continue

                for adj in self.adjacencies.get_compatible(tile_index, direction):
                    opposite_direction = Direction.opposite(direction)

                    enabler_counts = neighbor_cell.adjacency_tile_counts[adj]

                    if enabler_counts[opposite_direction.value] == 1:
                        if neighbor_cell.possible[adj]:
                            neighbor_cell.remove_possibility(adj)
                            if not any(neighbor_cell.possible):
                                raise RuntimeError("Hit a contradiction and cannot continue.")
                            heapq.heappush(self.entropy_heap, (neighbor_cell.entropy(), neighbor_index))
                            self.removal_stack.append((adj,neighbor_index))

                    neighbor_cell.adjacency_tile_counts[adj][opposite_direction.value] -= 1
                        
    def run(self):    
        while self.uncollapsed > 0:
            next_coord = self.choose_next_cell()
            self.collapse_cell_at(next_coord)
            self.propagate()
            self.uncollapsed -= 1