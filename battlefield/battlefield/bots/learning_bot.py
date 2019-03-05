import random
import numpy as np
from collections import defaultdict

from battlefield.bots.bot import Bot
from battlefield.avalon_types import filter_hidden_states, EVIL_ROLES, GOOD_ROLES, ProposeAction, MissionAction, VoteAction, PickMerlinAction, filter_hidden_states
from battlefield.bots.cfr_bot import EVIL_LOOKUP, MERLIN_LOOKUP, PROPOSAL_TO_INDEX_LOOKUP, proposal_to_bitstring, bitstring_to_proposal, INDEX_TO_PROPOSAL_2, INDEX_TO_PROPOSAL_3
from battlefield.bots.single_mcts_bot import heuristic_value_func

def reduce_base_n(arr, base):
    result = 0
    for i in range(len(arr)):
        result += (base**i)*arr[i]
    return result


def get_perspective(hidden_state, player):
    if hidden_state[player] == 'merlin':
        evil_1 = hidden_state.index('minion')
        evil_2 = hidden_state.index('assassin')

        bad_guy_index = EVIL_LOOKUP[5 * evil_1 + evil_2]
        assert bad_guy_index >= 0
        merlin_perspective = MERLIN_LOOKUP[10 * player + (bad_guy_index / 2)]
        assert merlin_perspective >= 0
        return merlin_perspective + 25
    elif hidden_state[player] in EVIL_ROLES:
        evil_1 = hidden_state.index('minion')
        evil_2 = hidden_state.index('assassin')

        partner = evil_2 if hidden_state[player] == 'minion' else evil_1
        perspective = EVIL_LOOKUP[5 * player + partner]
        assert perspective >= 0
        return perspective + 5
    else:
        return player


def history_to_bucket(hidden_state, player, history, player_status):
    _, current_state = history[-1]

    buck = player_status[:player] + player_status[player+1:]
    
    if current_state.status == 'merlin':
        return ('merlin', reduce_base_n(buck, 3))
    elif current_state.status == 'propose':
        size = current_state.MISSION_SIZES[current_state.succeeds + current_state.fails]
        return ('propose', get_perspective(hidden_state, player) * 81 * 2 + reduce_base_n(buck, 3) * 2 + int(size == 2))
    elif current_state.status == 'run':
        return ('run', 0)
    else:
        if current_state.propose_count == 4:
            return ('vote', get_perspective(hidden_state, player))
        proposal = tuple(sorted(list(current_state.proposal)))
        buck = reduce_base_n([player_status[p] for p in proposal], 3)

        if len(proposal) == 2:
            buck += 27

        bucket = (
            get_perspective(hidden_state, player) * 36 +
            buck
        )
        return ('vote', bucket + 55)


def move_index_to_move(move_index, state):
    if state.status == 'merlin':
        return PickMerlinAction(merlin=move_index)
    elif state.status == 'propose':
        size, _ = state.MISSION_SIZES[state.succeeds + state.fails]
        mapping = INDEX_TO_PROPOSAL_2 if size == 2 else INDEX_TO_PROPOSAL_3
        return ProposeAction(proposal=bitstring_to_proposal(mapping[move_index]))
    elif state.status == 'vote':
        return VoteAction(up=bool(move_index))
    else:
        return MissionAction(fail=bool(move_index))



C = 5
class Node:
    def __init__(self, num_moves):
        self.choose_counts = np.zeros(num_moves).astype(np.int)
        self.total_payoffs = np.zeros(num_moves)

    def select_move(self):
        if np.any(self.choose_counts == 0):
            p = (self.choose_counts == 0).astype(np.float)
            p /= float(np.sum(p))
            return np.random.choice(range(len(self.choose_counts)), p=p)
        average_payoff = self.total_payoffs / self.choose_counts

        total_choices = np.sum(self.choose_counts)
        weight_factor = average_payoff + C * np.sqrt(np.log(total_choices) / self.choose_counts)
        return np.argmax(weight_factor)


class LearningBot(Bot):
    def __init__(self):
        self.opponent_buckets = defaultdict(lambda: {
            'merlin': np.zeros((81, 5)).astype(np.int),
            'propose': np.zeros((55 * 81 * 2, 10)).astype(np.int),
            'run': np.zeros((1, 2)).astype(np.int),
            'vote': np.zeros((55 * 36 + 55, 2)).astype(np.int)
        })
        self.history = []
        self.my_buckets = {
            'merlin': defaultdict(lambda: Node(5)),
            'propose': defaultdict(lambda: Node(10)),
            'run': defaultdict(lambda: Node(2)),
            'vote': defaultdict(lambda: Node(2)),
        }
        self.game_num = 0


    def reset(self, game, player, role, hidden_states):
        self.history = [(None, game)]
        self.player = player
        self.hidden_states = hidden_states
        self.player_status = [0, 0, 0, 0, 0]
        self.fails = []
        self.role = role
        self.game_num += 1
        self.should_search = (self.game_num >= 100000) and (self.game_num % 10 == 0)
        if self.game_num == 100000:
            print "Searching from here on out..."


    def set_bot_ids(self, bot_ids):
        self.bot_ids = bot_ids


    def show_roles(self, hidden_state, bot_ids):
        player_statuses = [
            [0, 0, 0, 0, 0]
            for _ in hidden_state
        ]

        # update buckets and count
        for i in range(len(self.history) - 1):
            _, current_state = self.history[i]
            next_obs, _ = self.history[i+1]

            # Update the player_statuses
            if current_state.status == 'run':
                for player, player_status in enumerate(player_statuses):
                    p_on = player in current_state.proposal
                    if next_obs > 0:
                        for p in current_state.proposal:
                            player_status[p] = max(player_status[p], 2 if p_on else 1)

            # Get their moves
            moving_players = current_state.moving_players()
            if current_state.status == 'propose':
                moves = [PROPOSAL_TO_INDEX_LOOKUP[proposal_to_bitstring(next_obs)]]
            elif current_state.status == 'vote':
                moves = [int(vote.up) for vote in next_obs]
            elif current_state.status == 'run':
                evil_on_mission = [p for p in current_state.proposal if hidden_state[p] in EVIL_ROLES]
                failers = np.random.choice(evil_on_mission, size=next_obs, replace=False)
                moves = [1 if p in failers else 0 for p in current_state.proposal]
            elif current_state.status == 'merlin':
                merlin_index = hidden_state.index('merlin')
                if next_obs:
                    assassin_pick = merlin_index
                else:
                    assassin_pick = random.choice([p for p in range(len(hidden_state)) if p != merlin_index])
                moves = [ 0 if role != 'assassin' else assassin_pick for role in hidden_state]

            for player, move in zip(moving_players, moves):
                if current_state.status == 'run' and hidden_state[player] not in EVIL_ROLES:
                    continue
                if current_state.status == 'merlin' and hidden_state[player] != 'assassin':
                    continue
                bucket_type, bucket = history_to_bucket(hidden_state, player, self.history[:i+1], player_statuses[player])
                self.opponent_buckets[bot_ids[player]][bucket_type][bucket] += 1


    def handle_transition(self, old_state, new_state, observation, move=None):
        self.history.append((observation, new_state))
        if old_state.status == 'run':
            p_on = self.player in old_state.proposal
            if observation > 0:
                for p in old_state.proposal:
                    self.player_status[p] = max(self.player_status[p], 2 if p_on else 1)
                self.fails.append((old_state.proposal, observation))

            # filter hidden states
            if move is not None and self.role in EVIL_ROLES and not move.fail:
                observation += 1
            self.hidden_states = filter_hidden_states(self.hidden_states, old_state.proposal, observation)


    def single_mcts_search(self, state):
        history_len = len(self.history)
        hidden_state = self.hidden_states[np.random.choice(len(self.hidden_states))]
        player_statuses = [
            [0, 0, 0, 0, 0]
            for _ in hidden_state
        ]
        # Set up player statuses
        for proposal, _ in self.fails:
            for p in range(len(hidden_state)):
                p_on = p in proposal
                for player in proposal:
                    player_statuses[p][player] = max(player_statuses[p][player], 2 if p_on else 1)

        visited_nodes = []
        chosen_actions = []

        value = None
        while True:
            moves = []
            for player in state.moving_players():
                if hidden_state[player] not in EVIL_ROLES and state.status == 'run':
                    moves.append(MissionAction(fail=False))
                    continue

                if hidden_state[player] != 'assassin' and state.status == 'merlin':
                    moves.append(PickMerlinAction(merlin=np.random.choice(len(hidden_state))))
                    continue

                bucket_type, bucket = history_to_bucket(hidden_state, player, self.history, player_statuses[player])

                if player != self.player:
                    bucket_data = self.opponent_buckets[self.bot_ids[player]][bucket_type][bucket]

                    uniform_prob = np.ones(len(bucket_data))/len(bucket_data)
                    tremble_prob = 1.0/np.sqrt(np.sum(bucket_data) + 1)
                    if np.sum(bucket_data) == 0:
                        move_probs = uniform_prob
                    else:
                        move_probs = tremble_prob * uniform_prob + (1.0 - tremble_prob) * bucket_data / np.sum(bucket_data)

                    move_index = np.random.choice(len(move_probs), p=move_probs)
                else:
                    is_new_node = bucket not in self.my_buckets[bucket_type]
                    node = self.my_buckets[bucket_type][bucket]
                    move_index = node.select_move()
                    if is_new_node:
                        value = heuristic_value_func(state, hidden_state, self.player)
                    else:
                        visited_nodes.append(node)
                        chosen_actions.append(move_index)

                moves.append(move_index_to_move(move_index, state))

            if value is not None:
                break

            state, _, observation = state.transition(moves, hidden_state)
            self.history.append((observation, state))

            if state.is_terminal():
                value = state.terminal_value(hidden_state)[self.player]
                break


        for node, action in zip(visited_nodes, chosen_actions):
            node.choose_counts[action] += 1
            node.total_payoffs[action] += 1

        self.history = self.history[:history_len]


    def mcts_search(self, state, num_iterations=100):
        for _ in range(num_iterations):
            self.single_mcts_search(state)


    def get_action(self, state, legal_actions):
        if len(legal_actions) == 1:
            return legal_actions[0]

        if self.should_search:
            self.mcts_search(state, num_iterations=100)

        bucket_type, bucket = history_to_bucket(self.hidden_states[0], self.player, self.history, self.player_status)
        return move_index_to_move(self.my_buckets[bucket_type][bucket].select_move(), state)


    def get_move_probabilities(self, state, legal_actions):
        raise NotImplemented
