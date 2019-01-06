from collections import namedtuple, defaultdict
from proposal_game import ProposalGame, HiddenState, Round, Move

import numpy as np
import random
'''
Terminology:

    Hypothesis is a triple of (hidden state, explanation, score)
        The hidden state is the state which the score is conditioned on
        The explanation is the list of size rounds x players which contains
            all actions for each player at each timestep
        The score is logP(explanation | h) = sum logP(action | h)

    TOM is your theory of mind -- it is what you believe other people are
    thinking at any given time. 

        For a level 0 & level 1 player, the TOM is 
            empty. In our model, a level 0 player plays according to 
            their parity with probability always 1-eps, for eps ~= 0.005. 
            A level 0 player keeps no TOM, and thus a level 1 player also 
            has no TOM.
        For level >= 2, the TOM is a list of hypotheses for each player. These
            correspond to the hypothesis of the particles that the player
            has kept (and not pruned). From these hypotheses, we can reconstruct
            the belief of the player.


'''
Hypothesis = namedtuple('Hypothesis', ['h', 'explanation', 'score'])
TOM = namedtuple('TOM', ['thoughts'])
Particle = namedtuple('Particle', ['Hypothesis', 'TOM'])

EmptyTOM = TOM(None)
EPSILON = 0.005


# ------------ UTILS ------------

# Add and normalize arrays of log probabilities

def log_add(x, y):
    x = max(x, y)
    y = min(x, y)
    return x + y

def log_normalize_and_convert(arr):
    return np.exp(arr - max(arr)) / np.sum(np.exp(arr - max(arr)))


# ------------ AGENT ------------

class Agent(object):
    def __init__(self, game, level, my_starting_distribution, is_bad):
        self.game = game
        self.level = level      
        self.belief = my_starting_distribution
        # probably redundant but probably also useful
        self.is_bad = is_bad
        self.index = 0 # or something idk
        self.my_particles = self.generate_and_prune_new_particles(game, None, None, self.level, None)
        self.MCTS_NITER = 10
        self.THRESHOLD = -3000

    def _initial_particles(self, game, level, hidden_states=None):
        # hidden states is all valid hidden states
        if hidden_states is None:
            hidden_states = {i:game.HIDDEN_STATES[i] for i in range(len(game.HIDDEN_STATES))}
        if level == 0:
            return None

        if level == 1:
            return [
                Particle(
                    Hypothesis(hidden_state, [], np.log(self.belief[h])), EmptyTOM
                ) 
                for h, hidden_state in hidden_states.items()
            ]

        if level > 1:
            return [
                Particle(
                    Hypothesis(hidden_state, [], np.log(self.belief[h])),
                    TOM([
                        self._initial_particles(game, level-1)
                        for player in range(game.NUM_PLAYERS)
                    ])
                )
                for h, hidden_state in hidden_states.items()
            ]


    def get_belief_given_particles(self, game, particles):
        '''
        Returns the belief vector for that player over hidden states given a set of particles. Only 
        call this _after_ pruning the particles.

        Output: a H, belief vector given the particles we observed
        '''
        player_belief = -3000 * np.ones((len(game.HIDDEN_STATES),))
        for particle in particles:
            player_belief[particle.Hypothesis.h] = log_add(player_belief[particle.Hypothesis.h], particle.Hypothesis.score)
        player_belief = log_normalize_and_convert(player_belief)
        return player_belief


    def base_strategy(self, game, player, state, hidden_state):
        ''' level 0 strat ''' 
        if state.proposal is not None and player in state.proposal:
            if hidden_state.evil == player:
                return {Move(type='Pass', extra=None) : EPSILON, Move(type='Fail', extra=None) : 1. - EPSILON }
            else:
                return {Move(type='Pass', extra=None) : 1.}
        moves = game.possible_moves(player, state, hidden_state)
        return {move : 1. / len(moves) for move in moves}


    def move_probs_given_h(self, game, player, state, hidden_state, particles, level):
        '''
        Given a single player's belief vector, and hypothesis, get the move
        probabilities for that player if they were playing at level
        
        Return a dict of {action : proba}
        '''
        if level == 0:
            # what a level 0 player would do
            move_probs = self.base_strategy(game, player, state, hidden_state)
        if level > 0:
            # do random tree search
            move_probs = self.random_tree_search(game, player, state, hidden_state, particles, level)
        return move_probs


    def random_tree_search(self, game, player, state_, hidden_state, particles, level):
        '''
        Given an input game and player state, conduct a random tree search
        to find the optimal move probabilities for that player.

        This random tree search is unbelievably naive at the moment. For each round,
        we just pick a random action and then do a heavy playout, updating our rewards at each
        timestep. The only intelligent move is at the end, we select the final action based
        on our updated beliefs.

        Return a dict of {action : proba} 
        '''
        rewards = {act : 0.0 for act in game.possible_moves(player, state_, hidden_state)} 
        for _ in range(self.MCTS_NITER):
            state = state_
            is_first_move = True
            first_move = None
            best_particle_with_hidden_state = None
            for particle in particles:
                if particle.Hypothesis.h == hidden_state:
                    best_particle_with_hidden_state = particle
            if best_particle_with_hidden_state is None:
                continue
            while not game.state_is_final(state):
                if state.round != game.NUM_PLAYERS:
                    moves = []
                    for p in range(game.NUM_PLAYERS):
                        if p == player:
                            # I am smart mans
                            if level >= 2:
                                belief = self.get_belief_given_particles(game, best_particle_with_hidden_state.TOM.thoughts[p])
                                probs = self.get_move_probs_from_belief(
                                    game, p, state, belief, best_particle_with_hidden_state.TOM.thoughts[p], hidden_state, level-1
                                )
                                moves.append(max(probs, key=probs.get))
                            elif level == 1:
                                probs = self.base_strategy(game, p, state, hidden_state)
                                moves.append(max(probs, key=probs.get))
                        else:
                            if level >= 2:
                                # your opponent is level 1 (ish)
                                belief = self.get_belief_given_particles(game, best_particle_with_hidden_state.TOM.thoughts[p])
                                probs = self.get_move_probs_from_belief(
                                    game, p, state, belief, best_particle_with_hidden_state.TOM.thoughts[p], hidden_state, level-1
                                )
                                moves.append(max(probs, key=probs.get))
                            elif level == 1:
                                # your opponent is level 0
                                probs = self.base_strategy(game, p, state, hidden_state)
                                moves.append(max(probs, key=probs.get))

                    if is_first_move:
                        first_move = moves[player]
                        is_first_move = False
                else:
                    moves = []
                    for p in range(game.NUM_PLAYERS):
                        if p == hidden_state.evil:
                            moves.append(Move(type=None, extra=None))                           
                        elif p == player:
                            # I am smart mans
                            belief = self.get_belief_given_particles(game, particles)
                            moves.append(Move(type='Pick', extra=np.argmax(belief)))
                        else:
                            if level >= 2:
                                # your opponent is level 1 (ish)
                                belief = self.get_belief_given_particles(game, best_particle_with_hidden_state.TOM.thoughts[p])
                                moves.append(Move(type='Pick', extra=np.argmax(belief)))
                            elif level == 1:
                                # your opponent is level 0
                                moves.append(Move(type='Pick', extra=random.choice(range(game.NUM_PLAYERS))))
                if is_first_move or first_move is None:
                    first_move = moves[player]
                round_rewards = game.rewards(state, hidden_state, moves)
                rewards[first_move] += round_rewards[player]
                new_state = game.transition(state, hidden_state, moves)
                observation = game.observation(state, hidden_state, moves)
                new_particles = self.generate_and_prune_new_particles(game, state, observation, level, particles)
                state, particles = new_state, new_particles
        # TODO: this doesn't work
        rewards = {a : r - min(rewards.values()) + 1 for a, r in rewards.items()}
        reward_sum = sum(rewards.values())
        if reward_sum > 0:
            return {a : float(r + 1.)/float(reward_sum) for a, r in rewards.items()}
        else:
            return {a : 1./len(rewards) for a in rewards}


    def score_actions(self, game, prev_state, actions, observation, hidden_state, thm, level):
        '''
        Scores a new set of actions in a round given a previous state and a particle.
        Algorithm proceeds in 3 steps:
        1_  Get the belief of the player given your TOM for that player (specifically,)
            particle.TOM[player]
        2_  Get the move probabilities for that player given his belief and your particle
            hypothesis
        3_  Compute the score of his true action given the move probabilities you believe
            him to be operating under, and add it to the particle's existing score

        Return the overall score after accounting for all actions
        '''
        score = 0.
        for player, action in enumerate(actions):
            if level == 1:
                # TOM is empty, your opponent has no belief
                belief = None
                particles = None
            else:
                belief = self.get_belief_given_particles(game, thm.thoughts[player])
                particles = thm.thoughts[player]
            probs = self.get_move_probs_from_belief(
                game, player, prev_state, belief, particles, hidden_state, level-1
            )
            action_sets = game.infer_action_sets(prev_state, observation, hidden_state)
            moves = [action_sets[i][player] for i in range(len(action_sets))]
            for move in moves:
                if move not in probs:
                    probs[move] = EPSILON
                else:
                    probs[move] += EPSILON
            probs = {a : float(r) / sum(probs.values()) for a, r in probs.items()}
            if action in moves and probs[action] > 0:
                score += np.log(probs[action])
            else:
                return float('-inf')
        return score


    def update_TOM(self, game, thm, prev_state, actions, observation, level):
        if level == 0:
            assert False, "Should never have a TOM for level 0 player"

        if level == 1:
            return EmptyTOM

        if level > 1:
            particles = []
            for player in range(game.NUM_PLAYERS):
                new_particles = self.generate_and_prune_new_particles(game, prev_state, observation, level-1, thm.thoughts[player])
                particles.append(new_particles)
            return TOM(thoughts=particles)


    def generate_and_prune_new_particles(self, game, prev_state, observation, level, particles):
        if prev_state is None:
            if self.is_bad:
                # TODO: fix this to only take one particle if you know you're a baddie
                hidden_state = HIDDEN_STATE(evil=self.index)
                return self._initial_particles(game, level, hidden_states={self.index: hidden_state})
            return self._initial_particles(game, level)
        my_new_particles = []
        for particle in particles:
            action_sets = game.infer_action_sets(prev_state, observation, particle.Hypothesis.h)
            for actions in action_sets:
                if len(actions) == 0:
                    continue
                new_actions = particle.Hypothesis.explanation + [actions]
                new_TOM = self.update_TOM(game, particle.TOM, prev_state, actions, observation, level)
                new_score = log_add(particle.Hypothesis.score, self.score_actions(
                    game, prev_state, actions, observation, particle.Hypothesis.h, new_TOM, level
                ))
                if new_score < self.THRESHOLD:
                    if level == self.level:
                        print(actions, new_score)
                    continue
                new_hypothesis = Hypothesis(particle.Hypothesis.h, new_actions, new_score)
                new_particle = Particle(new_hypothesis, new_TOM)
                my_new_particles.append(new_particle)
        if len(my_new_particles) == 0 and level == self.level:
            # TODO: code to regenerate particle
            assert False, "we need to regenerate particles"
        seen = []
        for particle in my_new_particles:
            if particle not in seen:
                seen.append(particle)
            else:
                continue 
        particles = sorted(seen, key=lambda x: x.Hypothesis.score)
        if level == 1:
            particles = particles[:10]
        return particles

    def get_move_probs_from_belief(self, game, player, prev_state, belief, particles, hidden_state, level):
        if belief is None:
            return self.base_strategy(game, player, prev_state, hidden_state)
        conditioned_move_probs = [
            self.move_probs_given_h(game, player, prev_state, hidden_state, particles, level)  
            for hidden_state in game.HIDDEN_STATES
        ]
        acceptable_moves = None
        prob = 1.
        for h in range(len(game.HIDDEN_STATES)):
            if prob == 0:
                continue
            if acceptable_moves is None:
                acceptable_moves = set(conditioned_move_probs[h].keys())
                
            else:
                acceptable_moves &= set(conditioned_move_probs[h].keys())

        if acceptable_moves is None or len(acceptable_moves) == 0:
            return {}

        weighted_move_probs = {move : 1 for move in acceptable_moves}
        for move in weighted_move_probs:
            for h, move_probs in enumerate(conditioned_move_probs):
                if move in move_probs:
                    # weight by move prob and likelihood of hidden state
                    weighted_move_probs[move] *= belief[h] * move_probs[move]
        return weighted_move_probs

    def get_move(self, game, state):
        my_belief = self.get_belief_given_particles(game, self.my_particles)
        conditioned_move_probs = [
            self.move_probs_given_h(game, self.index, state, hidden_state, self.level)  
            for hidden_state in game.HIDDEN_STATES
        ]
        prob = 1.
        acceptable_moves = None
        for h in range(len(game.HIDDEN_STATES)):
            if prob == 0:
                continue
            if acceptable_moves is None:
                acceptable_moves = set(conditioned_move_probs[h].keys())
                
            else:
                acceptable_moves &= set(conditioned_move_probs[h].keys())

        if acceptable_moves is None or len(acceptable_moves) == 0:
            return {}

        weighted_move_probs = {move : 1 for move in acceptable_moves}
        for move in weighted_move_probs:
            for h, move_probs in enumerate(conditioned_move_probs):
                if move in move_probs:
                    # weight by move prob and likelihood of hidden state
                    weighted_move_probs[move] *= my_belief[h] * move_probs[move]
        return weighted_move_probs