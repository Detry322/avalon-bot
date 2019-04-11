#include <iostream>
#include <memory>
#include <vector>
#include <cstdint>
#include <cassert>
#include <array>

#include "lookahead.h"
#include "lookup_tables.h"

#include "Eigen/Core"

using namespace std;

#define NUM_PLAYERS 5
#define NUM_EVIL 2
#define NUM_VIEWPOINTS (1 + 6 + 4 + 4)
#define NUM_ASSIGNMENTS 60
#define NUM_PROPOSAL_OPTIONS 10
#define TREMBLE_VALUE 1e-25

typedef Eigen::Array<double, NUM_VIEWPOINTS, 1> ViewpointVector;
typedef Eigen::Array<double, NUM_VIEWPOINTS, NUM_PROPOSAL_OPTIONS> ProposeData;
typedef Eigen::Array<double, NUM_VIEWPOINTS, 2> VoteData;
typedef Eigen::Array<double, NUM_VIEWPOINTS, 2> MissionData;
typedef Eigen::Array<double, NUM_VIEWPOINTS, NUM_PLAYERS> MerlinData;

enum NodeType {
    PROPOSE,
    VOTE,
    MISSION,
    TERMINAL_MERLIN,
    TERMINAL_NO_CONSENSUS,
    TERMINAL_TOO_MANY_FAILS,
    TERMINAL_PROPOSE_NN
};

struct LookaheadNode {
    NodeType type;
    int num_succeeds;
    int num_fails;
    int proposer;
    int propose_count;
    uint32_t proposal;
    int merlin_pick;

    ViewpointVector reach_probs[NUM_PLAYERS];
    ViewpointVector counterfactual_values[NUM_PLAYERS];

    // std::vector<std::unique_ptr<double[][
    std::unique_ptr<ProposeData> propose_regrets;
    std::unique_ptr<ProposeData> propose_strategy;

    std::unique_ptr<std::array<VoteData, NUM_PLAYERS>> vote_regrets;
    std::unique_ptr<std::array<VoteData, NUM_PLAYERS>> vote_strategy;

    std::unique_ptr<std::array<MissionData, NUM_PLAYERS>> mission_regrets;
    std::unique_ptr<std::array<MissionData, NUM_PLAYERS>> mission_strategy;

    std::unique_ptr<std::array<MerlinData, NUM_PLAYERS>> merlin_regrets;
    std::unique_ptr<std::array<MerlinData, NUM_PLAYERS>> merlin_strategy;

    std::vector<uint32_t> fails;
    std::vector<std::unique_ptr<LookaheadNode>> children;

    LookaheadNode() = default;

    int round() const { return this->num_succeeds + this->num_fails; }

    static std::unique_ptr<LookaheadNode> RootProposal(int num_succeeds, int num_fails, int proposer, int propose_count) {
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

    static std::unique_ptr<LookaheadNode> CopyParent(const LookaheadNode& parent) {
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
};

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
        for (int i = 0; i < NUM_EVIL; i++) {
            auto new_child = LookaheadNode::CopyParent(*node);
            if (i == 0) {
                new_child->num_succeeds++;
            } else {
                new_child->num_fails++;
                new_child->fails.push_back(new_child->proposal);
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

void fill_reach_probabilities(LookaheadNode* node) {
    switch (node->type) {
    case PROPOSE: {
        int player = node->proposer;
        for (int proposal = 0; proposal < NUM_PROPOSAL_OPTIONS; proposal++) {
            auto& child = node->children[proposal];
            for (int i = 0; i < NUM_PLAYERS; i++) {
                if (i == player) continue;
                child->reach_probs[i] = node->reach_probs[i];
            }

            child->reach_probs[player] = node->reach_probs[player] * node->propose_strategy->col(proposal);
        }
    } break;
    case VOTE: {
        for (uint32_t vote_pattern = 0; vote_pattern < (1 << NUM_PLAYERS); vote_pattern++) {
            auto& child = node->children[vote_pattern];
            for (int player = 0; player < NUM_PLAYERS; player++) {
                int vote = (vote_pattern >> player) & 1;
                child->reach_probs[player] = node->reach_probs[player] * node->vote_strategy->at(player).col(vote);
            }
        }
    } break;
    case MISSION: {
        // TODO this is fancy
    } break;
    default: break;
    }
}

void calculate_strategy(LookaheadNode* node) {
    switch (node->type) {
    case PROPOSE: {
        // Initialize the node's memory
        auto& player_regrets = *(node->propose_regrets);
        auto& player_strategy = *(node->propose_strategy);
        player_strategy = player_regrets;

        auto sums = player_strategy.rowwise().sum();
        player_strategy.colwise() /= sums;
        player_strategy = player_strategy.unaryExpr([](double v) { return std::isfinite(v) ? v : 1.0/NUM_PROPOSAL_OPTIONS; });
        player_strategy = (1.0 - TREMBLE_VALUE) * player_strategy + TREMBLE_VALUE * ProposeData::Constant(1.0/NUM_PROPOSAL_OPTIONS);
    } break;
    case VOTE: {
        for (int i = 0; i < NUM_PLAYERS; i++) {
            auto& player_regrets = node->vote_regrets->at(i);
            auto& player_strategy = node->vote_strategy->at(i);

            player_strategy = player_regrets;
            auto sums = player_strategy.rowwise().sum();
            player_strategy.colwise() /= sums;
            player_strategy = player_strategy.unaryExpr([](double v) { return std::isfinite(v) ? v : 0.5; });
            player_strategy = (1.0 - TREMBLE_VALUE) * player_strategy + TREMBLE_VALUE * VoteData::Constant(0.5);
        }
    } break;
    case MISSION: {
        for (int i = 0; i < NUM_PLAYERS; i++) {
            if (((1 << i) & node->proposal) == 0) continue;
            auto& player_regrets = node->mission_regrets->at(i);
            auto& player_strategy = node->mission_strategy->at(i);

            player_strategy = player_regrets;
            auto sums = player_strategy.rowwise().sum();
            player_strategy.colwise() /= sums;
            player_strategy = player_strategy.unaryExpr([](double v) { return std::isfinite(v) ? v : 0.5; });
            player_strategy = (1.0 - TREMBLE_VALUE) * player_strategy + TREMBLE_VALUE * MissionData::Constant(0.5);
        }
    } break;
    case TERMINAL_MERLIN: {
        for (int i = 0; i < NUM_PLAYERS; i++) {
            auto& player_regrets = node->merlin_regrets->at(i);
            auto& player_strategy = node->merlin_strategy->at(i);

            player_strategy = player_regrets;
            auto sums = player_strategy.rowwise().sum();
            player_strategy.colwise() /= sums;
            player_strategy = player_strategy.unaryExpr([](double v) { return std::isfinite(v) ? v : 0.2; });
            player_strategy = (1.0 - TREMBLE_VALUE) * player_strategy + TREMBLE_VALUE * MerlinData::Constant(0.2);
        }
    } break;
    default: break;
    }

    fill_reach_probabilities(node);

    for (auto& child : node->children) {
        calculate_strategy(child.get());
    }
}

void test() {
    // cout << "Double: " << TREMBLE_VALUE  << endl;
    // cout << "Double: " << TREMBLE_VALUE*TREMBLE_VALUE  << endl;
    // cout << "Double: " << TREMBLE_VALUE*TREMBLE_VALUE*TREMBLE_VALUE  << endl;
    // cout << "Double: " << TREMBLE_VALUE*TREMBLE_VALUE*TREMBLE_VALUE*TREMBLE_VALUE  << endl;
    // cout << "Double: " << TREMBLE_VALUE*TREMBLE_VALUE*TREMBLE_VALUE*TREMBLE_VALUE*TREMBLE_VALUE  << endl;
    // for (int i = 0; i < 10; i++) {
    auto lookahead = create_avalon_lookahead(2, 2, 3, 3, 2);
    calculate_strategy(lookahead.get());
    // test_math(lookahead.get());
    // cout << "Lookahead count: " << count_lookahead(lookahead.get()) << " nodes." << endl;
    // cout << "Lookahead                 PROPOSE: " << count_lookahead_type(lookahead.get(), PROPOSE) << endl;
    // cout << "Lookahead                    VOTE: " << count_lookahead_type(lookahead.get(), VOTE) << endl;
    // cout << "Lookahead                 MISSION: " << count_lookahead_type(lookahead.get(), MISSION) << endl;
    // cout << "Lookahead                  MERLIN: " << count_lookahead_type(lookahead.get(), MERLIN) << endl;
    // cout << "Lookahead   TERMINAL_NO_CONSENSUS: " << count_lookahead_type(lookahead.get(), TERMINAL_NO_CONSENSUS) << endl;
    // cout << "Lookahead TERMINAL_TOO_MANY_FAILS: " << count_lookahead_type(lookahead.get(), TERMINAL_TOO_MANY_FAILS) << endl;
    // cout << "Lookahead  TERMINAL_MERLIN_PICKED: " << count_lookahead_type(lookahead.get(), TERMINAL_MERLIN_PICKED) << endl;
    // cout << "Lookahead     TERMINAL_PROPOSE_NN: " << count_lookahead_type(lookahead.get(), TERMINAL_PROPOSE_NN) << endl;  
    // }
}
