from proposal_game import ProposalGame
from solver import Solver

def main():
	print "Solving..."
	k = 3
	starting_belief_tensor = ProposalGame.get_starting_belief_tensor(k)
	solver = Solver(ProposalGame)
	solver.solve_for_player(0, ProposalGame.START_PHYSICAL_STATE, starting_belief_tensor)
	print "Done!"

if __name__ == "__main__":
	main()