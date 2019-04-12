#include "lookahead.h"
#include "lookup_tables.h"

int LookaheadNode::round() const {
    return this->num_succeeds + this->num_fails;
}

std::unique_ptr<LookaheadNode> LookaheadNode::RootProposal(int num_succeeds, int num_fails, int proposer, int propose_count) {
    auto result = std::make_unique<LookaheadNode>();
    result->type = PROPOSE;
    result->num_succeeds = num_succeeds;
    result->num_fails = num_fails;
    result->proposer = proposer;
    result->propose_count = propose_count;
    result->proposal = 0;
    result->merlin_pick = -1;
    return result;
}

std::unique_ptr<LookaheadNode> LookaheadNode::CopyParent(const LookaheadNode& parent) {
    auto result = std::make_unique<LookaheadNode>();
    result->type = parent.type;
    result->num_succeeds = parent.num_succeeds;
    result->num_fails = parent.num_fails;
    result->proposer = parent.proposer;
    result->propose_count = parent.propose_count;
    result->proposal = parent.proposal;
    result->merlin_pick = parent.merlin_pick;
    result->fails = parent.fails;
    return result;
}

void add_lookahead_children(const int depth, LookaheadNode* node) {
    switch (node->type) {
    case PROPOSE: {
        const int* index_to_proposal = (ROUND_TO_PROPOSE_SIZE[node->round()] == 2) ? INDEX_TO_PROPOSAL_2 : INDEX_TO_PROPOSAL_3;
        for (int i = 0; i < NUM_PROPOSAL_OPTIONS; i++) {
            auto new_child = LookaheadNode::CopyParent(*node);
            new_child->type = VOTE;
            new_child->proposal = index_to_proposal[i];
            add_lookahead_children(depth - 1, new_child.get());
            node->children.push_back(std::move(new_child));
        }
    } break;
    case VOTE: {
        for (int i = 0; i < (1 << NUM_PLAYERS); i++) {
            auto new_child = LookaheadNode::CopyParent(*node);
            new_child->proposer = (new_child->proposer + 1) % NUM_PLAYERS;

            if (__builtin_popcount(i) <= NUM_PLAYERS/2 ) {
                new_child->propose_count++;
                new_child->proposal = 0;

                // Vote fails
                if (new_child->propose_count == 5) {
                    new_child->type = TERMINAL_NO_CONSENSUS;
                } else if (depth == 0) {
                    new_child->type = TERMINAL_PROPOSE_NN;
                } else {
                    new_child->type = PROPOSE;
                }
            } else {
                // Vote passes
                new_child->propose_count = 0;
                new_child->type = MISSION;
            }

            add_lookahead_children(depth, new_child.get());
            node->children.push_back(std::move(new_child));
        }
    } break;
    case MISSION: {
        for (int i = 0; i < NUM_EVIL + 1; i++) {
            auto new_child = LookaheadNode::CopyParent(*node);
            if (i == 0) {
                new_child->num_succeeds++;
            } else {
                new_child->num_fails++;
                new_child->fails.push_back(std::make_pair(new_child->proposal, i));
            }
            if (new_child->num_fails == 3) {
                new_child->type = TERMINAL_TOO_MANY_FAILS;
            } else if (new_child->num_succeeds == 3) {
                new_child->type = TERMINAL_MERLIN;
            } else if (depth == 0) {
                new_child->type = TERMINAL_PROPOSE_NN;
            } else {
                new_child->type = PROPOSE;
            }
            add_lookahead_children(depth, new_child.get());
            node->children.push_back(std::move(new_child));
        }
    } break;
    default: break;
    }
}

void allocate_lookahead_vectors(LookaheadNode* node) {
    for (int i = 0; i < NUM_PLAYERS; i++) {
        node->reach_probs[i] = ViewpointVector::Constant(1.0);
        node->counterfactual_values[i].setZero();
    }

    switch (node->type) {
    case PROPOSE: {
        // Initialize the node's memory
        node->propose_regrets = std::make_unique<ProposeData>();
        node->propose_regrets->setZero();
        node->propose_strategy = std::make_unique<ProposeData>();
        node->propose_strategy->setZero();
    } break;
    case VOTE: {
        node->vote_regrets = std::make_unique<std::array<VoteData, NUM_PLAYERS>>();
        node->vote_strategy = std::make_unique<std::array<VoteData, NUM_PLAYERS>>();
        for (int i = 0; i < NUM_PLAYERS; i++) {
            node->vote_regrets->at(i).setZero();
            node->vote_strategy->at(i).setZero();
        }
    } break;
    case MISSION: {
        node->mission_regrets = std::make_unique<std::array<MissionData, NUM_PLAYERS>>();
        node->mission_strategy = std::make_unique<std::array<MissionData, NUM_PLAYERS>>();
        for (int i = 0; i < NUM_PLAYERS; i++) {
            node->mission_regrets->at(i).setZero();
            node->mission_strategy->at(i).setZero();
        }
    } break;
    case TERMINAL_MERLIN: {
        node->merlin_regrets = std::make_unique<std::array<MerlinData, NUM_PLAYERS>>();
        node->merlin_strategy = std::make_unique<std::array<MerlinData, NUM_PLAYERS>>();
        for (int i = 0; i < NUM_PLAYERS; i++) {
            node->merlin_regrets->at(i).setZero();
            node->merlin_strategy->at(i).setZero();
        }
    } break;
    default: break;
    }

    for (auto& child : node->children) {
        allocate_lookahead_vectors(child.get());
    }
}

std::unique_ptr<LookaheadNode> create_avalon_lookahead(
    const int num_succeeds,
    const int num_fails,
    const int proposer,
    const int propose_count,
    const int depth) {

    auto root_node = LookaheadNode::RootProposal(num_succeeds, num_fails, proposer, propose_count);
    add_lookahead_children(depth, root_node.get());
    allocate_lookahead_vectors(root_node.get());
    return root_node;
}

int count_lookahead_type(LookaheadNode* node, NodeType type) {
    int total_count = (node->type == type) ? 1 : 0;
    for (auto& child : node->children) {
        total_count += count_lookahead_type(child.get(), type);
    }
    return total_count;
}

int count_lookahead(LookaheadNode* node) {
    int total_count = 1;
    for (auto& child : node->children) {
        total_count += count_lookahead(child.get());
    }
    return total_count;
}
