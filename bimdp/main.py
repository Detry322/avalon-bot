from proposal_game import ProposalGame, HiddenState
from solver import Solver

def main():
	print "Solving..."
	k = 2
	player = 0
	hidden_state = HiddenState(evil=1)
	starting_belief = ProposalGame.initial_belief(player, hidden_state)
	starting_belief_tensor = ProposalGame.get_starting_belief_tensor(k)
	solver = Solver(ProposalGame)
	print solver.solve_for_player(player, starting_belief, ProposalGame.START_PHYSICAL_STATE, starting_belief_tensor)
	print "Done!"

if __name__ == "__main__":
	main()