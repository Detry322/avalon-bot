#include <iostream>
#include <vector>
#include "optionparser.h"

using namespace std;

enum optionIndex { UNKNOWN, HELP };

const option::Descriptor usage[] = {
  { UNKNOWN,   0,   "",    "",       option::Arg::None,       "USAGE: deeprole [options]\n\nOptions:"},
  { HELP,      0,   "h",    "help",   option::Arg::None,       "  \t-h, --help  \tPrint usage and exit." },
  { 0, 0, 0, 0, 0, 0 }
};

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

  return 0;
}
