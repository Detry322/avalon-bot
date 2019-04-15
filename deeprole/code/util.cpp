#include <vector>
#include <algorithm>
#include <sstream>

#include "util.h"

std::mt19937 rng;

// struct Initialization {
//     AssignmentProbs starting_probs;
//     ViewpointVector solution_values[NUM_PLAYERS];

//     int num_succeeds;
//     int num_fails;
//     int proposer;
//     int propose_count;
//     int iterations;
//     int wait_iterations;
//     std::string generate_start_technique;

//     std::string Stringify() const;
// };

void seed_rng() {
    std::random_device rd;
    std::array<int, std::mt19937::state_size> seed_data;
    std::generate_n(seed_data.data(), seed_data.size(), std::ref(rd));
    std::seed_seq seq(std::begin(seed_data), std::end(seed_data));
    rng = std::mt19937(seq);
}

std::string Initialization::Stringify() const {
    static Eigen::IOFormat CSVFmt(10, Eigen::DontAlignCols, ",", ",", "", "", "", "");
    std::stringstream stream;
    // Add starting data
    stream << depth << ",";
    stream << num_succeeds << ",";
    stream << num_fails << ",";
    stream << propose_count << ",";
    stream << proposer << ",";
    stream << iterations << ",";
    stream << wait_iterations << ",";
    stream << generate_start_technique << ",";
    stream << starting_probs.format(CSVFmt) << ",";
    for (int i = 0; i < NUM_PLAYERS; i++) {
        stream << solution_values[i].format(CSVFmt) << ((i < NUM_PLAYERS - 1) ? "," : "");
    }
    return stream.str();
}

void prepare_initialization(
    const int depth,
    const int num_succeeds,
    const int num_fails,
    const int propose_count,
    Initialization* init
) {

}

std::unique_ptr<LookaheadNode> lookahead_from_initialization(const Initialization& init) {
    return std::unique_ptr<LookaheadNode>(nullptr);
}

void run_initialization_with_cfr(
    const int iterations,
    const int wait_iterations,
    Initialization* init
) {

}
