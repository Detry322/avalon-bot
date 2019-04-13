#ifndef CFR_PLUS_H_
#define CFR_PLUS_H_

#include "./lookahead.h"

void calculate_strategy(LookaheadNode* node);
void calculate_counterfactual_values(LookaheadNode* node, const AssignmentProbs& starting_probs);

void cfr_plus(LookaheadNode* root);

#endif // CFR_PLUS_H_
