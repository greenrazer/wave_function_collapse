import numpy as np
from PIL import Image

class ImageBuilder:
    def __init__(self, tile_grid, tile_size, output_size, mode):
        self.tile_index_grid = tile_grid
        self.output_size = output_size
        self.tile_size = tile_size
        self.mode = mode

        if mode == 'RGB':
            self.pixel_length = 3
        else:
            self.pixel_length = 4

    def to_file_paste_pixels(self, output_filename, pixels):
        output = np.zeros((np.prod(self.output_size), self.pixel_length))  

        for coord, tile_index in enumerate(self.tile_index_grid):
            output[coord] = pixels[tile_index]
        
        output = output.reshape(self.output_size + (self.pixel_length,)).astype(np.uint8)

        img = Image.fromarray(output)
        img.save(output_filename)
    
    def to_file_paste_tiles(self, output_filename, tiles):
        output = Image.new(self.mode, (self.output_size[0]*self.tile_size[0] + self.output_size[1]*self.tile_size[1]))

        for coord, tile_index in enumerate(self.tile_index_grid):
            x = coord % self.output_size[0]
            y = coord // self.output_size[0]
            output.paste(tiles[tile_index], (x*self.tile_size[0],y*self.tile_size[1]))

        output.save(output_filename)
        