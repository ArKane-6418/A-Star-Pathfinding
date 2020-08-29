import pygame
from queue import PriorityQueue
from typing import List, Tuple, Dict

# Setting up Pygame window
pygame.init()
WIDTH = 800
WINDOW = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("A* Pathfinding with Dijkstra's Algorithm")


# Colours

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
TEAL = (45, 247, 230)  # Start
DARK_BLUE = (0, 4, 252)  # End
RED = (255, 0, 0)  # Visited
GREEN = (0, 255, 0)
PURPLE = (207, 103, 214)  # Path
GREY = (128, 128, 128)


# Fonts

TITLE_FONT = pygame.font.SysFont("Times New Roman", 40)
TEXT_FONT = pygame.font.SysFont("Times New Roman", 30)

class Square:
    def __init__(self, row: int, col: int, width: int, total_rows: int) -> None:
        self.row = row
        self.col = col
        self.x = row * width
        self.y = col * width
        self.colour = WHITE
        self.neighbours = []
        self.width = width
        self.total_rows = total_rows

    def get_pos(self) -> Tuple[int, int]:
        return self.row, self.col

    def is_closed(self) -> bool:
        return self.colour == RED

    def is_open(self) -> bool:
        return self.colour == GREEN

    def is_barrier(self) -> bool:
        return self.colour == BLACK

    def is_start(self) -> bool:
        return self.colour == TEAL

    def is_end(self) -> bool:
        return self.colour == DARK_BLUE

    def clear(self) -> None:
        self.colour = WHITE

    def make_closed(self) -> None:
        self.colour = RED

    def make_open(self) -> None:
        self.colour = GREEN

    def make_barrier(self) -> None:
        self.colour = BLACK

    def make_start(self) -> None:
        self.colour = TEAL

    def make_end(self) -> None:
        self.colour = DARK_BLUE

    def make_path(self) -> None:
        self.colour = PURPLE

    def draw_square(self, window) -> None:
        pygame.draw.rect(window, self.colour,
                         (self.x, self.y, self.width, self.width))

    def update_neighbours(self, grid) -> None:
        self.neighbours = []
        if self.row < self.total_rows-1 and not grid[self.row + 1][self.col].is_barrier():
            self.neighbours.append(grid[self.row + 1][self.col])
        if self.row > 0 and not grid[self.row - 1][self.col].is_barrier():
            self.neighbours.append(grid[self.row - 1][self.col])
        if self.col < self.total_rows-1 and not grid[self.row][self.col + 1].is_barrier():
            self.neighbours.append(grid[self.row][self.col + 1])
        if self.col > 0 and not grid[self.row][self.col - 1].is_barrier():
            self.neighbours.append(grid[self.row][self.col - 1])

    def __lt__(self, other):
        return False


# Heuristic Function


def heuristic(point1: Tuple[int, int], point2: Tuple[int, int]) -> int:
    """Return the Manhattan (Taxicab) distance between two points"""
    x1, y1 = point1
    x2, y2 = point2

    return abs(x1 - x2) + abs(y1 - y2)


# Dijkstra's Algorithm


def dijkstra_algorithm(draw, grid: List[List[Square]], start: Square,
                       end: Square) -> bool:
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {}
    g_score = {square: float("inf") for row in grid for square in row}
    g_score[start] = 0
    f_score = {square: float("inf") for row in grid for square in row}
    f_score[start] = heuristic(start.get_pos(), end.get_pos())
    open_set_hash = {start}

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        current_node = open_set.get()[2]
        open_set_hash.remove(current_node)

        if current_node == end:
            reconstruct_path(came_from, end, draw)
            start.make_start()
            end.make_end()
            return True

        for neighbour in current_node.neighbours:
            # We assume the distance of each edge is 1
            temp_g_score = g_score[current_node] + 1
            if temp_g_score < g_score[neighbour]:
                came_from[neighbour] = current_node
                g_score[neighbour] = temp_g_score
                f_score[neighbour] = g_score[neighbour] + \
                                     heuristic(neighbour.get_pos(),
                                               end.get_pos())
                if neighbour not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbour], count, neighbour))
                    open_set_hash.add(neighbour)
                    neighbour.make_open()

        draw()

        if current_node != start:
            current_node.make_closed()

    return False


def reconstruct_path(came_from: Dict[Square, Square], current: Square, draw) -> None:
    length = 0
    while current in came_from:
        current = came_from[current]
        length += 1
        current.make_path()
        draw()
    print(f"The length of the shortest path between the start and end is {length} squares.")


def create_grid(width: int, rows: int) -> List[List[Square]]:
    """Create the grid with all the Square objects"""
    gap = width // rows
    grid = []

    for i in range(rows):
        grid.append([])
        for j in range(rows):
            square = Square(i, j, gap, rows)
            grid[i].append(square)
    return grid


def draw_grid_lines(window, width: int, rows: int) -> None:
    gap = width // rows

    for i in range(rows):
        pygame.draw.line(window, GREY, (0, i * gap), (width, i * gap))
    for j in range(rows):
        pygame.draw.line(window, GREY, (j * gap, 0), (j * gap, width))


def start_screen(window) -> None:
    window.fill(WHITE)
    title_text = TITLE_FONT.render("A* Pathfinding Algorithm", 1, BLACK)
    text = TEXT_FONT.render("Press any key to continue", 1, BLACK)
    window.blit(title_text, (WIDTH / 2 - (title_text.get_width() // 2), 40))
    window.blit(text, (WIDTH // 2 - (text.get_width() // 2), 300))
    pygame.display.update()
    wait_for_key()


def wait_for_key():
    clock = pygame.time.Clock()
    wait = True
    while wait:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                wait = False
                pygame.quit()
            if event.type == pygame.KEYDOWN:
                wait = False


def draw(window, grid: List[List[Square]], width: int, rows: int):
    window.fill(WHITE)
    for row in grid:
        for square in row:
            square.draw_square(window)
    draw_grid_lines(window, width, rows)
    pygame.display.update()


def get_clicked_position(pos: Tuple[int, int], width: int, rows: int) -> \
        Tuple[int, int]:
    gap = width // rows
    y, x = pos
    row = y // gap
    col = x // gap
    return row, col


def main(window, width):
    num_rows = 50
    grid = create_grid(width, num_rows)
    start = None  # Start node
    end = None  # End node
    run = True

    start_screen(window)

    while run:
        draw(window, grid, width, num_rows)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_position(pos, width, num_rows)
                square = grid[row][col]
                if not start and square != end:
                    start = square
                    start.make_start()
                elif not end and square != start:
                    end = square
                    end.make_end()

                elif square != end and square != start:
                    square.make_barrier()

            elif pygame.mouse.get_pressed()[2]:
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_position(pos, width, num_rows)
                square = grid[row][col]
                square.clear()
                if square == start:
                    start = None
                elif square == end:
                    end = None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and start and end:
                    for row in grid:
                        for square in row:
                            square.update_neighbours(grid)
                    dijkstra_algorithm(lambda: draw(window, grid, width, num_rows),
                                       grid, start, end)
                if event.key == pygame.K_c:
                    start = None
                    end = None
                    grid = create_grid(width, num_rows)

    pygame.quit()


main(WINDOW, WIDTH)
