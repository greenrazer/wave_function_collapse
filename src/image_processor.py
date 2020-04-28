from PIL import Image
from src.direction import Direction
from src.frequency_rules import FrequencyRules
from src.adjacency_rules import AdjacencyRules

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