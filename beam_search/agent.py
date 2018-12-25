from collections import namedtuple, defaultdict

import searcher
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
	def __init__(self, game, level, my_starting_distribution, is_bad):
		self.game = game
		self.level = level		
		self.belief = my_starting_distribution
		# probably redundant but probably also useful
		self.is_bad = is_bad
		self.index = 0 # or something idk
		self.my_particles = generate_and_prune_new_particles(self, game, None, None, None, None)
		self.MCTS_NITER = 1000

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
		player_belief = np.zeros((len(game.HIDDEN_STATES),))
		for particle in player_particles:
			player_belief[particle.Hypothesis.h] = log_add(player_belief[particle.Hypothesis.h], particle.score)
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


	def move_probs_given_h(self, game, player, state, hidden_state, level):
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
			move_probs = self.random_tree_search(game, player, state, hidden_state)
		return move_probs


	def random_tree_search(self, game, player, state, hidden_state):
		'''
		Given an input game and player state, conduct a random tree search
		to find the optimal move probabilities for that player.

		This random tree search is unbelievably naive at the moment. For each round,
		we just pick a random action and then do a heavy playout, updating our rewards at each
		timestep. The only intelligent move is at the end, we select the final action based
		on our updated beliefs.

		Return a dict of {action : proba} 
			'''
		rewards = {act : 0 for act in game.possible_moves(player, state, hidden_state)}
		for _ in range(self.MCTS_NITER):
			is_first_move = True
			while !game.state_is_final(state):
				if state.round != game.NUM_PLAYERS:
					moves = [
						random.choice(game.possible_moves(p, state, hidden_state)) 
						for p in game.NUM_PLAYERS
					]
					if is_first_move:
						first_move = moves[player]
						is_first_move = False
				else:
					moves = []
					for p in range(game.NUM_PLAYERS):
						if p == hidden_state.evil:
							moves.append(Move(type=None, extra=None))							
						elif p == player:
							belief = self.get_belief_given_particles(game, particles)
							moves.append(Move(type='Pick', extra=np.argmax(belief)))
						else:
							belief = self.get_belief_given_particles(game, particles[0].TOM[p])
							moves.append(Move(type='Pick', extra=np.argmax(belief)))
				round_rewards = game.rewards(state, hidden_state, moves)
				rewards[first_move] += round_rewards[player]
				new_state = game.transition(state, hidden_state, moves)
				new_particles = self.generate_and_prune_new_particles(game, state, moves, level, particles)
				state, particles = new_state, new_particles
		rewards = {a : max(0, r) for a, r in rewards.items()}
		reward_sum = sum(rewards.values())
		return {a : r / reward_sum for a, r in rewards.items()}

	def score_actions(self, game, prev_state, actions, hidden_state, TOM, level):
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
			if level == 1:
				# TOM is empty, your opponent has no belief
				belief = None
			else:
				belief = self.get_beliefs_given_particles(game, TOM[player])
			probs = self.get_move_probs_from_belief(
				game, player, prev_state, belief, hidden_state, level-1
			)
			score = log_add(score, np.log(probs[action])) if action in probs else float('-inf')
		return log_add(score, particle.score)


	def update_TOM(self, game, TOM, prev_state, actions, level):
		if level == 0:
			assert False, "Should never have a TOM for level 0 player"

		if level == 1:
			return EmptyTOM

		if level > 1:
			for player in range(game.NUM_PLAYERS):
				new_particles = self.generate_and_prune_new_particles(game, prev_state, actions, level-1, TOM.thoughts)
				return TOM(new_particles)


	def generate_and_prune_new_particles(self, game, prev_state, actions, level, particles):
		if prev_state is None:
			if self.is_bad:
				# TODO: fix this to only take one particle if you know you're a baddie
				hidden_state = HIDDEN_STATE(evil=self.index)
				return self._initial_particles(game, level, hidden_states={self.index: hidden_state})
			return self._initial_particles(game, level)
		my_new_particles = []
		for particle in particles:
			new_actions = particle.Hypothesis.explanation + actions
			new_TOM = self.updateTOM(game, particle.TOM, prev_state, actions, level)
			new_score = score_actions(self, game, prev_state, actions, new_hypothesis.h, new_TOM, level)
			if new_score < self.THRESHOLD:
				continue
			new_hypothesis = Hypothesis(particle.Hypothesis.h, new_actions, new_score)
			new_particle = Particle(new_hypothesis, new_TOM)
			my_new_particles.append(new_particle)

		if len(my_new_particles) == 0:
			#self.regenerate_all_particles()
			assert False, "let's make em from scratch"
		return my_new_particles


	def get_move(self, game, state):
		my_belief = self.get_belief_given_particles(game, self.my_particles)
		conditioned_move_probs = [
			self.move_probs_given_h(game, self.index, state, hidden_state, self.level)	
			for hidden_state in game.HIDDEN_STATES
		]
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