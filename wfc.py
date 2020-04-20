import numpy as np
import argparse
import random
import enum
import heapq
import copy
from PIL import Image
from typing import List

class StartFromZeroEnum(enum.Enum):
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        return count

class Direction(StartFromZeroEnum):
    LEFT = enum.auto()
    UP = enum.auto()
    RIGHT = enum.auto()
    DOWN = enum.auto()

    @staticmethod
    def opposite(dir: 'Direction') -> 'Direction':
        return Direction((dir.value + 2) % 4)

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

class ImageProcessor:
    def __init__(self, filename, tile_size=(3,3), rotation=False):
        self.image = Image.open(filename)
        self.tile_size = tile_size
        self.rotation = rotation

        self.tiles = []
        self.top_left_pixels = []

        self.adjacencies = AdjacencyRules()
        self.frequencies = FrequencyRules()

        self.process_image()
    
    def process_image(self):
        self.get_frequencies()
        self.get_adjacencys()
    
    def get_frequencies(self):
        if self.rotation:
            imgs = [self.image, self.image.transpose(Image.ROTATE_90), self.image.transpose(Image.ROTATE_180), self.image.transpose(Image.ROTATE_270)]
        else:
            imgs = [self.image]
        
        tile_to_ind = {}
        for img in imgs:
            for x in range(0, img.width - self.tile_size[0] + 1):
                for y in range(0, img.height - self.tile_size[1] + 1):
                    cropped_im = img.crop((x,y,x+self.tile_size[0], y+self.tile_size[1]))
                    im_hash = cropped_im.tobytes()
                    if im_hash not in tile_to_ind:
                        self.tiles.append(cropped_im)
                        self.top_left_pixels.append(cropped_im.getpixel((0,0)))
                        tile_to_ind[im_hash] = len(self.tiles)-1
                    self.frequencies.add(tile_to_ind[im_hash])
    
    def compatible(self, tile1_ind:int, tile2_ind:int, direction:Direction) -> bool:
        if direction == Direction.UP:
            return self.tiles[tile1_ind].crop((0,0, self.tile_size[0], self.tile_size[1]-1)) == self.tiles[tile2_ind].crop((0,1, self.tile_size[0], self.tile_size[1]))
        elif direction == Direction.DOWN:
            return self.tiles[tile1_ind].crop((0,1, self.tile_size[0], self.tile_size[1])) == self.tiles[tile2_ind].crop((0,0, self.tile_size[0], self.tile_size[1]-1))
        elif direction == Direction.LEFT:
            return self.tiles[tile1_ind].crop((0,0, self.tile_size[0]-1, self.tile_size[1])) == self.tiles[tile2_ind].crop((1,0, self.tile_size[0], self.tile_size[1]))
        elif direction == Direction.RIGHT:
            return self.tiles[tile1_ind].crop((1,0, self.tile_size[0], self.tile_size[1])) == self.tiles[tile2_ind].crop((0,0, self.tile_size[0]-1, self.tile_size[1]))

    def get_adjacencys(self):
        for t1 in range(len(self.tiles)):
            for t2 in range(len(self.tiles)):
                for direction in Direction:
                    if self.compatible(t1, t2, direction):
                        self.adjacencies.add(t1, t2, direction)

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


def main(input_filename, output_filename, tile_size, output_size, rotate):
    image_processor = ImageProcessor(input_filename, tile_size, rotate)

    keep_going = True
    while keep_going:
        try:
            wfc_state = WFCState(
                len(image_processor.tiles), 
                image_processor.adjacencies, 
                image_processor.frequencies, 
                output_size)
            wfc_state.run()
            keep_going = False
        except RuntimeError:
            print("Contradiction! Restarting...")
    
    if image_processor.image.mode == 'RGB':
        pix_len = 3
    else:
        pix_len = 4

    output = np.zeros((np.prod(output_size), pix_len))    
    for coord, cell in enumerate(wfc_state.grid):
        tile_index = [i for i, x in enumerate(cell.possible) if x][0]
        output[coord] = image_processor.top_left_pixels[tile_index]
    output = output.reshape(output_size + (pix_len,)).astype(np.uint8)

    img = Image.fromarray(output)
    img.save(output_filename)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Wave function colapse algorithm.')
    parser.add_argument('input_filename', type=str, help='Input Filename.')
    parser.add_argument('output_filename', type=str, help='Output Filename.')
    parser.add_argument('-tw', '--tile_width', type=int, default=3, help='Tile size width.')
    parser.add_argument('-th', '--tile_height', type=int, default=3, help='Tile size height.')
    parser.add_argument('-ow', '--output_width', type=int, default=20, help='Output size width.')
    parser.add_argument('-oh', '--output_height', type=int, default=20, help='Output size height.')
    parser.add_argument('-r', '--rotate', default=False, action='store_true', help="Use all rotated input image tiles.")
    parser.add_argument('-s', '--seed', type=int, default=0, help='Output size height.')
    args = parser.parse_args()

    if args.seed:
        random.seed(args.seed)

    main(args.input_filename, args.output_filename, (args.tile_width, args.tile_height), (args.output_width, args.output_height), args.rotate)
