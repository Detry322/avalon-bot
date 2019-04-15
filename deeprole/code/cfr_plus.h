#ifndef CFR_PLUS_H_
#define CFR_PLUS_H_

#include "./lookahead.h"

void calculate_strategy(LookaheadNode* node);
void calculate_counterfactual_values(LookaheadNode* node, const AssignmentProbs& starting_probs);

void cfr_get_values(
    LookaheadNode* root,
    const int iterations,
    const int wait_iterations,
    const AssignmentProbs& starting_probs,
    ViewpointVector* values
);

#endif // CFR_PLUS_H_
