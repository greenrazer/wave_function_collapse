import enum

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