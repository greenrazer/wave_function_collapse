import argparse
import numpy as np
import random
from PIL import Image

from src.image_processor import ImageProcessor
from src.image_builder import ImageBuilder
from src.wfc_state import WFCState as WFC

def main(input_filename, output_filename, tile_size, output_size, rotate):
    image_processor = ImageProcessor(input_filename, tile_size, rotate)

    keep_going = True
    while keep_going:
        try:
            wfc_state = WFC(
                len(image_processor.tiles), 
                image_processor.adjacencies, 
                image_processor.frequencies, 
                output_size)
            wfc_state.run()
            keep_going = False
        except RuntimeError:
            print("Contradiction! Restarting...")

    tile_grid = [0 for _ in range(np.prod(output_size))]    
    for coord, cell in enumerate(wfc_state.grid):
        tile_grid[coord] = [i for i, x in enumerate(cell.possible) if x][0]
    
    image_builder = ImageBuilder(tile_grid, tile_size, output_size, image_processor.image.mode)
    image_builder.to_file_paste_pixels(output_filename, image_processor.top_left_pixels)
    
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