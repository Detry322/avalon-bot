from mcts_common import simulate, determinization_iterator
from moismcts import Node, select_leaf, expand_if_needed, backpropagate


def search_mtmoismcts(initial_game_state, possible_hidden_states, num_iterations):
    all_roots = [
        [ Node(parent=None, incoming_edge=None), Node(parent=None, incoming_edge=None) ]
        for player in range(initial_game_state.NUM_PLAYERS)
    ]

    for i, initial_hidden_state in determinization_iterator(possible_hidden_states, num_iterations):
        roots = [ all_roots[player][initial_hidden_state == player] for player in range(initial_game_state.NUM_PLAYERS) ]

        determinization = [(initial_game_state, initial_hidden_state)]
        nodes, determinization = select_leaf(roots, determinization)
        nodes, determinization = expand_if_needed(nodes, determinization)
        rewards = simulate(*determinization[-1])
        backpropagate(nodes, rewards, determinization)

    root = all_roots[0][0]
    return max(root.children, key=lambda action: root.children[action].visit_count), all_roots


if __name__ == "__main__":
    from proposal_game import ProposalGameState
    start = ProposalGameState.start_state()
    possible_hidden_states = [1, 2]
    action, all_roots = search_mtmoismcts(start, possible_hidden_states, 100000)
    print action
    print {
        node.incoming_edge: (
            node.visit_count,
            node.total_reward / node.visit_count
        )
        for node in all_roots[0][0].children.values()
    }

