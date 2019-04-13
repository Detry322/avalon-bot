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

using namespace std;

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

static void add_middle_cfvs(LookaheadNode* node, int me, int my_viewpoint, int my_partner, double* pass_cfv, double* fail_cfv) {
    assert(node->type == MISSION);
    assert(VIEWPOINT_TO_BAD[me][my_viewpoint] == my_partner);
    assert(my_viewpoint >= NUM_GOOD_VIEWPOINTS);
    int partner_viewpoint = VIEWPOINT_TO_PARTNER_VIEWPOINT[me][my_viewpoint];
    assert(VIEWPOINT_TO_PARTNER_VIEWPOINT[my_partner][partner_viewpoint] == my_viewpoint);
    assert(node->proposal & (1 << me));
    assert(node->proposal & (1 << my_partner));

    double partner_responsibility = my_single_pass_responsibility(node, my_partner, partner_viewpoint, me);
    double middle_cfv = node->children[1]->counterfactual_values[me](my_viewpoint);
    middle_cfv /= partner_responsibility;
    double partner_pass_prob = node->mission_strategy->at(my_partner)(partner_viewpoint, 0);
    *pass_cfv += middle_cfv * partner_pass_prob;
    *fail_cfv += middle_cfv * (1.0 - partner_pass_prob);
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

static void populate_full_reach_probs(LookaheadNode* node) {
    for (int i = 0; i < NUM_ASSIGNMENTS; i++) {
        double probability = 1.0;
        int evil = ASSIGNMENT_TO_EVIL[i];
        for (auto fail : node->fails) {
            if (__builtin_popcount(fail.first & evil) < fail.second) {
                probability = 0.0;
                break;
            }
        }
        for (int player = 0; player < NUM_PLAYERS && probability != 0; player++) {
            int viewpoint = ASSIGNMENT_TO_VIEWPOINT[i][player];
            probability *= node->reach_probs[player](viewpoint);
        }
        (*(node->full_reach_probs))(i) = probability;
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
        // Intentional missing break.
    }
    case TERMINAL_NO_CONSENSUS:
    case TERMINAL_TOO_MANY_FAILS:
    case TERMINAL_PROPOSE_NN: {
        populate_full_reach_probs(node);
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

    // Update regrets
    for (int proposal = 0; proposal < NUM_PROPOSAL_OPTIONS; proposal++) {
        node->propose_regrets->col(proposal) = node->children[proposal]->counterfactual_values[node->proposer] - node->counterfactual_values[node->proposer];
    }
    *(node->propose_regrets) = node->propose_regrets->max(0.0);
}

static void calculate_vote_cfvs(LookaheadNode* node) {
    for (int player = 0; player < NUM_PLAYERS; player++) {
        VoteData cfvs = VoteData::Constant(0.0);

        for (int vote_pattern = 0; vote_pattern < (1 << NUM_PLAYERS); vote_pattern++) {
            auto& child = node->children[vote_pattern];
            int vote = (vote_pattern >> player) & 1;
            cfvs.col(vote) += child->counterfactual_values[player];
        }

        // Update regrets
        node->counterfactual_values[player] = (cfvs * node->vote_strategy->at(player)).rowwise().sum();
        node->vote_regrets->at(player) += cfvs.colwise() - node->counterfactual_values[player];
        node->vote_regrets->at(player) = node->vote_regrets->at(player).max(0.0);
    }
}

static void calculate_mission_cfvs(LookaheadNode* node) {
    // For players not on the mission, the CFVs are just the sum.
    for (int player = 0; player < NUM_PLAYERS; player++) {
        // Skip players on the mission
        if ((1 << player) & node->proposal) continue;

        for (int num_fails = 0; num_fails < NUM_EVIL + 1; num_fails++) {
            node->counterfactual_values[player] += node->children[num_fails]->counterfactual_values[player];
        }
    }

    // For players on the mission, the CFVs are a little more complicated.
    for (int player = 0; player < NUM_PLAYERS; player++) {
        // Skip players not on the mission.
        if (((1 << player) & node->proposal) == 0) continue;

        // For good viewpoints, CFVs are just the sum of the number of possible fails
        for (int viewpoint = 0; viewpoint < NUM_GOOD_VIEWPOINTS; viewpoint++) {
            for (int num_fails = 0; num_fails < NUM_EVIL + 1; num_fails++) {
                node->counterfactual_values[player](viewpoint) += node->children[num_fails]->counterfactual_values[player](viewpoint);
            }   
        }

        // For bad viewpoints, CFVs are split.
        for (int viewpoint = NUM_GOOD_VIEWPOINTS; viewpoint < NUM_VIEWPOINTS; viewpoint++) {
            double pass_cfv = 0.0;
            double fail_cfv = 0.0;
            int partner = VIEWPOINT_TO_BAD[player][viewpoint];
            if ((1 << partner) & node->proposal) {
                // The partner is on the mission.
                pass_cfv = node->children[0]->counterfactual_values[player](viewpoint);
                fail_cfv = node->children[2]->counterfactual_values[player](viewpoint);
                add_middle_cfvs(node, player, viewpoint, partner, &pass_cfv, &fail_cfv);
            } else {
                // The partner is not on the mission. CFVs are "simple" - 0 or 1 fails possible.
                assert(node->children[2]->counterfactual_values[player](viewpoint) == 0.0);
                pass_cfv = node->children[0]->counterfactual_values[player](viewpoint);
                fail_cfv = node->children[1]->counterfactual_values[player](viewpoint);
            }
            double my_pass_prob = node->mission_strategy->at(player)(viewpoint, 0);
            double result_cfv = pass_cfv * my_pass_prob + fail_cfv * (1.0 - my_pass_prob);
            node->mission_regrets->at(player)(viewpoint, 0) += pass_cfv - result_cfv;
            node->mission_regrets->at(player)(viewpoint, 1) += fail_cfv - result_cfv;
        }
        node->mission_regrets->at(player) = node->mission_regrets->at(player).max(0.0);
    }
}

static void calculate_merlin_cfvs(LookaheadNode* node, const AssignmentProbs& starting_probs) {
    for (int i = 0; i < NUM_ASSIGNMENTS; i++) {
        int merlin = ASSIGNMENT_TO_ROLES[i][0];
        int assassin = ASSIGNMENT_TO_ROLES[i][1];
        int assassin_viewpoint = ASSIGNMENT_TO_VIEWPOINT[i][assassin];
        int evil = ASSIGNMENT_TO_EVIL[i];
        double reach_prob = (*(node->full_reach_probs))(i) * starting_probs(i);
        if (reach_prob == 0.0) continue;

        double correct_prob = node->merlin_strategy->at(assassin)(assassin_viewpoint, merlin);

        for (int player = 0; player < NUM_PLAYERS; player++) {
            int viewpoint = ASSIGNMENT_TO_VIEWPOINT[i][player];
            double counterfactual_reach_prob = reach_prob / node->reach_probs[player](viewpoint);
            if ((1 << player) & evil) {
                node->counterfactual_values[player](viewpoint) += counterfactual_reach_prob * (
                    EVIL_WIN_PAYOFF * correct_prob +
                    EVIL_LOSE_PAYOFF * (1.0 - correct_prob)
                );
            } else {
                node->counterfactual_values[player](viewpoint) += counterfactual_reach_prob * (
                    GOOD_LOSE_PAYOFF * correct_prob +
                    GOOD_WIN_PAYOFF * (1.0 - correct_prob)
                );
            }
        }

        double assassin_counterfactual_reach_prob = reach_prob / node->reach_probs[assassin](assassin_viewpoint);
        double expected_assassin_payoff = assassin_counterfactual_reach_prob * (
            EVIL_WIN_PAYOFF * correct_prob +
            EVIL_LOSE_PAYOFF * (1.0 - correct_prob)
        );
        for (int assassin_choice = 0; assassin_choice < NUM_PLAYERS; assassin_choice++) {
            // double choice_prob = node->merlin_strategy->at(assassin)(assassin_viewpoint, assassin_choice);
            double payoff = (
                (assassin_choice == merlin) ?
                (EVIL_WIN_PAYOFF * assassin_counterfactual_reach_prob) :
                (EVIL_LOSE_PAYOFF * assassin_counterfactual_reach_prob)
            );

            node->merlin_strategy->at(assassin)(assassin_viewpoint, assassin_choice) += payoff - expected_assassin_payoff;
        }
    }

    for (int player = 0; player < NUM_PLAYERS; player++) {
        node->merlin_strategy->at(player) = node->merlin_strategy->at(player).max(0.0);
    }
};

static void calculate_terminal_cfvs(LookaheadNode* node, const AssignmentProbs& starting_probs) {
    for (int i = 0; i < NUM_ASSIGNMENTS; i++) {
        int evil = ASSIGNMENT_TO_EVIL[i];
        double reach_prob = (*(node->full_reach_probs))(i) * starting_probs(i);
        for (int player = 0; player < NUM_PLAYERS; player++) {
            int viewpoint = ASSIGNMENT_TO_VIEWPOINT[i][player];
            double counterfactual_reach_prob = reach_prob / node->reach_probs[player](viewpoint);
            if ((1 << player) & evil) {
                node->counterfactual_values[player](viewpoint) += counterfactual_reach_prob * EVIL_WIN_PAYOFF; // In these terminal nodes, evil wins, good loses.
            } else {
                node->counterfactual_values[player](viewpoint) += counterfactual_reach_prob * GOOD_LOSE_PAYOFF;
            }
        }
    }
}

static void calculate_neural_net_cfvs(LookaheadNode* node, const AssignmentProbs& starting_probs) {
    assert(false); // No support for neural net yet.
};

void calculate_counterfactual_values(LookaheadNode* node, const AssignmentProbs& starting_probs) {
    for (auto& child : node->children) {
        calculate_counterfactual_values(child.get(), starting_probs);
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
        calculate_merlin_cfvs(node, starting_probs); break;
    case TERMINAL_NO_CONSENSUS:
    case TERMINAL_TOO_MANY_FAILS:
        calculate_terminal_cfvs(node, starting_probs); break;
    case TERMINAL_PROPOSE_NN:
        calculate_neural_net_cfvs(node, starting_probs); break;
    }
}

void cfr_plus(LookaheadNode* root) {
    AssignmentProbs starting_probs = AssignmentProbs::Constant(1.0/NUM_ASSIGNMENTS);
    for (int i = 0; i < 10; i++) { 
        calculate_strategy(root);
        calculate_counterfactual_values(root, starting_probs);
        cout << "Iteration " << i << " " << root->counterfactual_values[0].transpose() << endl;
    }
}

