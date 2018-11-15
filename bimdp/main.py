from proposal_game import ProposalGame, HiddenState
from solver import Solver

import numpy as np

def get_move_and_expected_payoff(belief, action_probs):
	move = max(action_probs, key=lambda move: action_probs[move][0])
	reward = sum(a * b if a != 0 else 0 for a, b in zip(belief, action_probs[move][1]))
	return move, reward

def play_game():
	k = 5
	hidden_state = HiddenState(evil=1)
	beliefs = [ProposalGame.initial_belief(player, hidden_state) for player in range(ProposalGame.NUM_PLAYERS)]
	belief_tensor = ProposalGame.get_starting_belief_tensor(k)
	solver = Solver(ProposalGame)
	state = ProposalGame.START_PHYSICAL_STATE
	for player in range(ProposalGame.NUM_PLAYERS):
		print "Solving player {}".format(player)
		solver.solve_for_player(player, beliefs[player], state, belief_tensor)


	total_rewards = np.zeros(ProposalGame.NUM_PLAYERS)

	print "================= HIDDEN STATE: {}".format(hidden_state)
	while not ProposalGame.state_is_final(state):
		print "========= STATE: {}".format(state)

		# Get moves
		action_probs_for_players = [
			solver.solve_for_player(player, beliefs[player], state, belief_tensor)
			for player in range(ProposalGame.NUM_PLAYERS)
		]

		moves_and_expected_payoffs = [
			get_move_and_expected_payoff(beliefs[player], action_probs)
			for player, action_probs in enumerate(action_probs_for_players)
		]
		moves, expected_rewards = zip(*moves_and_expected_payoffs)
		for player in range(ProposalGame.NUM_PLAYERS):
			print "=== Player {}".format(player)
			print " Belief: {}".format(beliefs[player])
			print " Move  : {}".format(moves[player])
			print " Expects a reward of: {}".format(expected_rewards[player])
		print "====="
		# Make moves
		next_state = ProposalGame.transition(state, hidden_state, moves)
		observation = ProposalGame.observation(state, hidden_state, moves)
		rewards = ProposalGame.rewards(state, hidden_state, moves)
		total_rewards += rewards

		print " Observation: {}".format(observation)
		print " Rewards: {}".format(rewards)
		print " Total R: {}".format(total_rewards)

		for player in range(ProposalGame.NUM_PLAYERS):
			beliefs[player] = solver.single_update(player, beliefs[player], moves[player], belief_tensor, state, next_state, observation)
			print "New beliefs {}: {}".format(player, beliefs[player])

		belief_tensor = solver.update_tensor(belief_tensor, state, next_state, observation)

		state = next_state

def main():
	play_game()

if __name__ == "__main__":
	main()