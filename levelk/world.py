import copy
import itertools

MOVE_UP = 0
MOVE_RIGHT = 1
MOVE_DOWN = 2
MOVE_LEFT = 3
STAY = 4

ACTIONS = [MOVE_UP, MOVE_RIGHT, MOVE_DOWN, MOVE_LEFT, STAY]

def load_world(filename, num_players):
    data = [line.strip() for line in open(filename)]
    return World(data, num_players)

def _set_loc(players, player, new_loc):
    return players[:player] + (new_loc, ) + players[player+1:]

PLAYER_NAME = '0123456789abcdef'

class World(object):
    def __init__(self, data, num_players):
        self.players = (None,)*num_players
        self.height = len(data)
        self.width = len(data[0])
        self.obstacles = set([])
        self.goals = set([])
        for i in range(self.height):
            for j in range(self.width):
                if data[i][j] in PLAYER_NAME:
                    self.players = _set_loc(self.players, PLAYER_NAME.index(data[i][j]), (i, j))
                elif data[i][j] == 'X':
                    self.obstacles.add((i, j))
                elif data[i][j] == 'G':
                    self.goals.add((i, j))
        self.obstacles = frozenset(self.obstacles)
        self.goals = frozenset(self.goals)


    def is_game_end(self):
        return any([player in self.goals for player in self.players])


    def _next_pos(self, location, move):
        i, j = location
        if move == MOVE_UP:
            i -= 1
        elif move == MOVE_DOWN:
            i += 1
        elif move == MOVE_LEFT:
            j -= 1
        elif move == MOVE_RIGHT:
            j += 1
        elif move == STAY:
            pass

        i = max(0, min(self.height - 1, i))
        j = max(0, min(self.width - 1, j))
        if (i, j) in self.obstacles:
            return location
        return (i, j)


    def next_state(self, moves):
        next_players = tuple([self._next_pos(loc, mov) for loc, mov in zip(self.players, moves)])
        for p1 in range(len(moves)):
            for p2 in range(len(moves)):
                if next_players[p1] == next_players[p2] and p1 != p2:
                    next_players = _set_loc(next_players, p1, self.players[p1])
                    next_players = _set_loc(next_players, p2, self.players[p2])

        new_world = copy.copy(self)
        new_world.players = next_players
        rewards = [0.0 if move == STAY else -1.0 for move in moves]
        for player, new_pos in enumerate(next_players):
            if new_pos in self.goals:
                rewards[player] += 11.0
        return new_world, rewards


    def all_reachable_states(self):
        explored_states = set([])
        stack = [self]
        while len(stack) != 0:
            to_explore = stack.pop()
            if to_explore in explored_states:
                continue
            explored_states.add(to_explore)

            for moves in itertools.product(ACTIONS, repeat=len(self.players)):
                new_state, _ = to_explore.next_state(moves)
                if new_state not in explored_states:
                    stack.append(new_state)

        return list(explored_states)



    def __eq__(self, other):
        return self.players == other.players and self.height == other.height and self.width == other.width and self.obstacles == other.obstacles and self.goals == other.goals

    def __hash__(self):
        hash_val = (self.players, self.obstacles, self.goals, self.height, self.width)
        return hash(hash_val)


    def __str__(self):
        result = "WORLD:\n"
        for i in range(self.height):
            row = ""
            for j in range(self.width):
                loc = (i, j)
                if loc in self.players:
                    row += str(self.players.index(loc))
                elif loc in self.obstacles:
                    row += "X"
                elif loc in self.goals:
                    row += 'G'
                else:
                    row += "."
            result += row + "\n"
        return result

    def __repr__(self):
        return self.__str__()
