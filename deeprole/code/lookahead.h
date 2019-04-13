#ifndef LOOKAHEAD_H_
#define LOOKAHEAD_H_

#include <memory>
#include <array>
#include <vector>
#include "Eigen/Core"

#include "lookup_tables.h"

typedef Eigen::Array<double, NUM_VIEWPOINTS, 1> ViewpointVector;
typedef Eigen::Array<double, NUM_VIEWPOINTS, NUM_PROPOSAL_OPTIONS> ProposeData;
typedef Eigen::Array<double, NUM_VIEWPOINTS, 2> VoteData;
typedef Eigen::Array<double, NUM_VIEWPOINTS, 2> MissionData;
typedef Eigen::Array<double, NUM_VIEWPOINTS, NUM_PLAYERS> MerlinData;
typedef Eigen::Array<double, NUM_ASSIGNMENTS, 1> AssignmentProbs;

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
    
    std::unique_ptr<AssignmentProbs> full_reach_probs;

    std::unique_ptr<ProposeData> propose_regrets;
    std::unique_ptr<ProposeData> propose_strategy;

    std::unique_ptr<std::array<VoteData, NUM_PLAYERS>> vote_regrets;
    std::unique_ptr<std::array<VoteData, NUM_PLAYERS>> vote_strategy;

    std::unique_ptr<std::array<MissionData, NUM_PLAYERS>> mission_regrets;
    std::unique_ptr<std::array<MissionData, NUM_PLAYERS>> mission_strategy;

    std::unique_ptr<std::array<MerlinData, NUM_PLAYERS>> merlin_regrets;
    std::unique_ptr<std::array<MerlinData, NUM_PLAYERS>> merlin_strategy;

    std::vector<std::pair<uint32_t, int>> fails;
    std::vector<std::unique_ptr<LookaheadNode>> children;

    LookaheadNode() = default;

    int round() const;

    static std::unique_ptr<LookaheadNode> RootProposal(int num_succeeds, int num_fails, int proposer, int propose_count);
    static std::unique_ptr<LookaheadNode> CopyParent(const LookaheadNode& parent);
};

void add_lookahead_children(const int depth, LookaheadNode* node);
void allocate_lookahead_vectors(LookaheadNode* node);
std::unique_ptr<LookaheadNode> create_avalon_lookahead(
    const int num_succeeds,
    const int num_fails,
    const int proposer,
    const int propose_count,
    const int depth);

int count_lookahead_type(LookaheadNode* node, const NodeType type);
int count_lookahead(LookaheadNode* node);

#endif // LOOKAHEAD_H_
