from collections import namedtuple, defaultdict
import numpy as np
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
	return x + np.log(1 + np.exp(y - x))

def log_normalize_and_convert(arr):
	return np.exp(arr - max(arr)) / np.sum(np.exp(arr - max(arr)))

# ------------ AGENT ------------

class Agent(object):
	def __init__(self, game, level, my_starting_distribution):
		self.game = game
		self.level = level		
		self.belief = my_starting_distribution
		self.my_particles = self._initial_particles(game, level)
		self.my_particles = self.prune_particles(game)

	def _initial_particles(self, game, level):
		if level == 0:
			return None

		if level == 1:
			return [Particle(Hypothesis(hidden_state, [], np.log(self.belief[h])), EmptyTOM) for h, hidden_state in enumerate(game.HIDDEN_STATES)]

		if level > 1:
			return [
				Particle(
					Hypothesis(hidden_state, [], np.log(self.belief[h])),
					TOM([
						self._initial_particles(game, level-1)
						for player in range(game.NUM_PLAYERS)
					])
				)
				for h, hidden_state in enumerate(game.HIDDEN_STATES)
			]


	def get_belief_given_particles(self, game, particles):
		'''
		Returns the belief vector for that player over hidden states given a set of particles. Only 
		call this _after_ pruning the particles.

		Output: a H, belief vector given the particles we observed
		'''
		player_belief = np.zeros((len(game.HIDDEN_STATES),))
		for particle in player_particles:
			player_belief[particle.h] = log_add(player_belief[particle.Hypothesis.h], particle.score)
		player_belief = log_normalize_and_convert(player_belief)
		return player_belief


	def base_strategy(self, player, state, hidden_state):
		''' level 0 strat ''' 
        if state.proposal is not None and player in state.proposal:
            if hidden_state.evil == player:
                return {Move(type='Pass', extra=None) : EPSILON, Move(type='Fail', extra=None) : 1. - EPSILON }
            else:
                return {Move(type='Pass', extra=None) : 1.}
        moves = cls.possible_moves(player, state, hidden_state)
        return {move : 1. / len(moves) for move in moves}


	def get_move_probs_from_belief(self, game, player, state, belief, hypothesis, level):
		'''
		Given a single player's belief vector, and hypothesis, get the move
		probabilities for that player if they were playing at level
		
		Return a dict of {action : proba}
		'''
		if level == 0:
			# what a level 0 player would do
			actions = self.base_strategy(player, state, hypothesis.h)




	def score_actions(self, game, prev_state, actions, particle):
		'''
		Scores a new set of actions in a round given a previous state and a particle.
		Algorithm proceeds in 3 steps:
		1_  Get the belief of the player given your TOM for that player (specifically,)
		   	particle.TOM[player]
		2_	Get the move probabilities for that player given his belief and your particle
			hypothesis
		3_ 	Compute the score of his true action given the move probabilities you believe
			him to be operating under, and add it to the particle's existing score

		Return the overall score after accounting for all actions
		'''
		score = 0
		for player, action in enumerate(actions):
			belief = self.get_beliefs_given_particles(game, particle.TOM[player])
			probs = self.get_move_probs_from_belief(
				game, player, prev_state, belief, particle.Hypothesis, self.level-1
			)
			score += np.log(probs[action]) if action in probs else float('-inf')
		return score + particle.score


	def generate_new_particles(self, game, prev_state, actions):
		my_new_particles = []
		for particle in self.my_particles:
			new_score = score_actions(self, game, prev_state, actions, particle)
			#TODO: fast-path this to prune here, instead of separate function.



	def prune_particles(self, game):
		my_new_particles = []
		for particle in self.my_particles:
			if particle.score > self.THRESHOLD:
				my_new_particles.append(particle)
		if len(my_new_particles) == 0:
			#self.regenerate_all_particles()
			assert False, "let's make em from scratch"
		self.my_particles = my_new_particles


	def get_move(self, game):
		pass
