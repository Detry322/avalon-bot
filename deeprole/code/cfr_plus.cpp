#include <iostream>
#include <memory>
#include <vector>
#include <cstdint>
#include <cassert>
#include <cmath>
#include <array>
#include <utility>

#include "cfr_plus.h"
#include "lookup_tables.h"

static double my_single_pass_responsibility(LookaheadNode* node, int me, int my_viewpoint, int my_partner) {
    assert(node->type == MISSION);
    assert(VIEWPOINT_TO_BAD[me][my_viewpoint] == my_partner);
    assert(my_viewpoint >= NUM_GOOD_VIEWPOINTS);
    int partner_viewpoint = VIEWPOINT_TO_PARTNER_VIEWPOINT[me][my_viewpoint];
    assert(VIEWPOINT_TO_PARTNER_VIEWPOINT[my_partner][partner_viewpoint] == my_viewpoint);
    assert(node->proposal & (1 << me));
    assert(node->proposal & (1 << my_partner));
    double my_pass_prob = node->mission_strategy->at(me)(my_viewpoint, 0);
    double partner_pass_prob = node->mission_strategy->at(my_partner)(partner_viewpoint, 0);
    double outcome_prob = my_pass_prob * (1.0 - partner_pass_prob) + (1.0 - my_pass_prob) * partner_pass_prob;
    double my_responsibility_portion = my_pass_prob * my_pass_prob + (1.0 - my_pass_prob) * (1.0 - my_pass_prob);
    double partner_responsibility_portion = partner_pass_prob * partner_pass_prob + (1.0 - partner_pass_prob) * (1.0 - partner_pass_prob);
    double my_responsibility_exponent = my_responsibility_portion / (my_responsibility_portion + partner_responsibility_portion);
    return pow(outcome_prob, my_responsibility_exponent);
}

static void fill_reach_probabilities_for_mission_node(LookaheadNode* node) {
    assert(node->type == MISSION);

    for (int fails = 0; fails < NUM_EVIL + 1; fails++) {
        auto& child = node->children[fails];
        // First, pass-through all of the players not on the mission.
        // Second, pass-through all of the viewpoints where the player is good.
        //      - this step won't work on it's own, we need to fix things at the leaf node to ensure we remove possibilities where a good player was forced to fail.
        // This is optimized - we'll be multiplying "correctly" later down in this function.
        for (int player = 0; player < NUM_PLAYERS; player++) {
            child->reach_probs[player] = node->reach_probs[player];
        }

        switch (fails) {
        case 0: {
            // No one failed.
            // For evil players, their move probability gets multiplied by the pass probability.
            for (int player = 0; player < NUM_PLAYERS; player++) {
                // Skip players not on the mission.
                if (((1 << player) & node->proposal) == 0) continue;
                for (int viewpoint = NUM_GOOD_VIEWPOINTS; viewpoint < NUM_VIEWPOINTS; viewpoint++) {
                    child->reach_probs[player](viewpoint) *= node->mission_strategy->at(player)(viewpoint, 0);
                }
            }
        } break;
        case 1: {
            // This is the hard case, since we're combining probabilities oddly.
            for (int player = 0; player < NUM_PLAYERS; player++) {
                // Skip players not on the mission.
                if (((1 << player) & node->proposal) == 0) continue;
                for (int viewpoint = NUM_GOOD_VIEWPOINTS; viewpoint < NUM_VIEWPOINTS; viewpoint++) {
                    int player_partner = VIEWPOINT_TO_BAD[player][viewpoint];
                    if ((1 << player_partner) & node->proposal) {
                        // The player's partner is on the mission. Weird stuff!
                        child->reach_probs[player](viewpoint) *= my_single_pass_responsibility(node, player, viewpoint, player_partner);
                    } else {
                        // The player's partner is not on the mission, fail normally.
                        child->reach_probs[player](viewpoint) *= node->mission_strategy->at(player)(viewpoint, 1);
                    }
                }
            }
        } break;
        case 2: {
            // Everyone failed.
            // For evil players, their move probability gets multiplied by the fail probability.
            for (int player = 0; player < NUM_PLAYERS; player++) {
                // Skip players not on the mission.
                if (((1 << player) & node->proposal) == 0) continue;
                for (int viewpoint = NUM_GOOD_VIEWPOINTS; viewpoint < NUM_VIEWPOINTS; viewpoint++) {
                    child->reach_probs[player](viewpoint) *= node->mission_strategy->at(player)(viewpoint, 1);
                }
            }
        } break;
        }
    }
}

static void fill_reach_probabilities(LookaheadNode* node) {
    switch (node->type) {
    case PROPOSE: {
        int player = node->proposer;
        for (int proposal = 0; proposal < NUM_PROPOSAL_OPTIONS; proposal++) {
            auto& child = node->children[proposal];
            for (int i = 0; i < NUM_PLAYERS; i++) {
                child->reach_probs[i] = node->reach_probs[i];
            }

            child->reach_probs[player] *= node->propose_strategy->col(proposal);
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
        fill_reach_probabilities_for_mission_node(node);
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

static void calculate_propose_cfvs(LookaheadNode* node) {
    for (int player = 0; player < NUM_PLAYERS; player++) {
        for (int proposal = 0; proposal < NUM_PROPOSAL_OPTIONS; proposal++) {
            auto& child = node->children[proposal];
            if (player == node->proposer) {
                node->counterfactual_values[player] += child->counterfactual_values[player] * node->propose_strategy->col(proposal);
            } else {
                node->counterfactual_values[player] += child->counterfactual_values[player];
            }
        }
    }
}

static void calculate_vote_cfvs(LookaheadNode* node) {
    for (int player = 0; player < NUM_PLAYERS; player++) {
        for (int vote_pattern = 0; vote_pattern < (1 << NUM_PLAYERS); vote_pattern++) {
            auto& child = node->children[vote_pattern];
            int vote = (vote_pattern >> player) & 1;
            node->counterfactual_values[player] += child->counterfactual_values[player] * node->vote_strategy->at(player).col(vote);
        }
    }
}

static void calculate_mission_cfvs(LookaheadNode* node) {
    // todo
}

static void calculate_merlin_cfvs(LookaheadNode* node) {
    // todo
};

static void calculate_terminal_cfvs(LookaheadNode* node) {
    // todo
}

static void calculate_neural_net_cfvs(LookaheadNode* node) {
    assert(false); // No support for neural net yet.
};

void calculate_counterfactual_values(LookaheadNode* node) {
    for (auto& child : node->children) {
        calculate_counterfactual_values(child.get());
    }

    for (int player = 0; player < NUM_PLAYERS; player++) {
        node->counterfactual_values[player].setZero();
    }

    switch (node->type) {
    case PROPOSE:
        calculate_propose_cfvs(node); break;
    case VOTE:
        calculate_vote_cfvs(node); break;
    case MISSION:
        calculate_mission_cfvs(node); break;
    case TERMINAL_MERLIN:
        calculate_merlin_cfvs(node); break;
    case TERMINAL_NO_CONSENSUS:
    case TERMINAL_TOO_MANY_FAILS:
        calculate_terminal_cfvs(node); break;
    case TERMINAL_PROPOSE_NN:
        calculate_neural_net_cfvs(node); break;
    }
}

void cfr_plus(LookaheadNode* root) {
    calculate_strategy(root);
    calculate_counterfactual_values(root);
}
