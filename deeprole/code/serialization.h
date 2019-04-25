#ifndef SERIALIZATION_H_
#define SERIALIZATION_H_

#include <iostream>

#include "./lookahead.h"

void json_deserialize_starting_reach_probs(std::istream& in_stream, AssignmentProbs* starting_reach_probs);
void json_serialize_lookahead(const LookaheadNode* root, const AssignmentProbs& starting_reach_probs, std::ostream& out_stream);

#endif // SERIALIZATION_H_
