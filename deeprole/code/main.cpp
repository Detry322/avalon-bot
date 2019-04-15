#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include "optionparser.h"
#include "lookahead.h"
#include "cfr_plus.h"
#include "util.h"

using namespace std;

enum optionIndex { UNKNOWN, HELP };

const option::Descriptor usage[] = {
    { UNKNOWN,   0,   "",    "",       option::Arg::None,       "USAGE: deeprole [options]\n\nOptions:"},
    { HELP,      0,   "h",    "help",   option::Arg::None,       "  \t-h, --help  \tPrint usage and exit." },
    { 0, 0, 0, 0, 0, 0 }
};

std::string random_string(std::string::size_type length)
{
    static auto& chrs = "0123456789abcdef";
    static std::uniform_int_distribution<std::string::size_type> pick(0, sizeof(chrs) - 2);

    std::string s;

    s.reserve(length);

    while(length--)
        s += chrs[pick(rng)];

    return s;
}

void print_lookahead_information(const int depth, const int num_succeeds, const int num_fails, const int propose_count) {
    auto lookahead = create_avalon_lookahead(num_succeeds, num_fails, 0, propose_count, depth);
    cout << "                PROPOSE: " << count_lookahead_type(lookahead.get(), PROPOSE) << endl;
    cout << "                   VOTE: " << count_lookahead_type(lookahead.get(), VOTE) << endl;
    cout << "                MISSION: " << count_lookahead_type(lookahead.get(), MISSION) << endl;
    cout << "        TERMINAL_MERLIN: " << count_lookahead_type(lookahead.get(), TERMINAL_MERLIN) << endl;
    cout << "  TERMINAL_NO_CONSENSUS: " << count_lookahead_type(lookahead.get(), TERMINAL_NO_CONSENSUS) << endl;
    cout << "TERMINAL_TOO_MANY_FAILS: " << count_lookahead_type(lookahead.get(), TERMINAL_TOO_MANY_FAILS) << endl;
    cout << "    TERMINAL_PROPOSE_NN: " << count_lookahead_type(lookahead.get(), TERMINAL_PROPOSE_NN) << endl;
    cout << "                  Total: " << count_lookahead(lookahead.get()) << endl;
}

void generate_datapoints(
    const int num_datapoints,
    const int depth,
    const int num_succeeds,
    const int num_fails,
    const int propose_count,
    const int iterations,
    const int wait_iterations,
    const std::string output_dir,
    const std::string filename_suffix
) {
    std::string base_filename = (
        "d" + std::to_string(depth) + "_" +
        "s" + std::to_string(num_succeeds) + "_" +
        "f" + std::to_string(num_fails) + "_" +
        "p" + std::to_string(propose_count) + "_" +
        "i" + std::to_string(iterations) + "_" +
        "w" + std::to_string(wait_iterations) + "_" +
        ((filename_suffix.empty()) ? random_string(16) : filename_suffix) +
        ".csv"
    );
    const std::string filepath = ((output_dir.empty()) ? "" : (output_dir + "/")) + base_filename;

    cout << "=========== DEEPROLE DATAPOINT GENERATOR =========" << endl;
    cout << "           # Datapoints: " << num_datapoints << endl;
    cout << "           # Iterations: " << iterations << endl;
    cout << "           # Wait iters: " << wait_iterations << endl;
    cout << "------------------ Game settings -------------------" << endl;
    cout << "                  Depth: " << depth << endl;
    cout << "                  Round: " << (num_succeeds + num_fails) << endl;
    cout << "               Succeeds: " << num_succeeds << endl;
    cout << "                  Fails: " << num_fails << endl;
    cout << "              Propose #: " << propose_count << endl;
    cout << "------------------ Sanity checks -------------------" << endl;
    print_lookahead_information(depth, num_succeeds, num_fails, propose_count);
    cout << "------------------ Administration ------------------" << endl;
    cout << " Output directory: " << "'" << output_dir << "'" << endl;
    cout << " Writing to: " << filepath << endl;
    cout << "====================================================" << endl;

    const int status_interval = max(1, min(100, num_datapoints/10));

    std::fstream fs;
    fs.open(filepath, std::fstream::out | std::fstream::app);

    for (int i = 0; i < num_datapoints; i++) {
        if (i % status_interval == 0) {
            cout << i << "/" << num_datapoints << endl;
        }

        Initialization init;
        prepare_initialization(depth, num_succeeds, num_fails, propose_count, &init);
        run_initialization_with_cfr(iterations, wait_iterations, &init);

        fs << init.Stringify() << endl;
    }
    fs.close();
}

void test() {
    auto lookahead = create_avalon_lookahead(2, 2, 3, 4, 2);
    AssignmentProbs starting_probs = AssignmentProbs::Constant(1.0/NUM_ASSIGNMENTS);
    ViewpointVector values[NUM_PLAYERS];
    cfr_get_values(lookahead.get(), 3000, 1000, starting_probs, values);
    cout << values[0].transpose() << endl;
}

int main(int argc, char* argv[]) {
    argc -= (argc > 0); argv += (argc > 0); // skip program name argv[0] if present
    
    option::Stats stats(usage, argc, argv);
    std::vector<option::Option> options(stats.options_max);
    std::vector<option::Option> buffer(stats.buffer_max);
    option::Parser parse(usage, argc, argv, &options[0], &buffer[0]);
    if (parse.error())
        return 1;

    if (options[HELP]) {
        option::printUsage(std::cout, usage);
        return 0;
    }
    seed_rng();
    generate_datapoints(10, 1, 2, 2, 4, 3000, 1000, "deeprole_output", "");
    // test();
    return 0;
}
