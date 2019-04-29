#include "./nn.h"

#include <map>
#include <sstream>
#include <iostream>

#include "eigen_types.h"
#include "lookup_tables.h"
#include "game_constants.h"

std::map<std::tuple<int, int, int>, std::shared_ptr<Model>> model_cache;

static std::string get_model_filename(const std::string& search_dir, const int num_succeeds, const int num_fails, const int propose_count) {
    std::stringstream sstream;
    sstream << ((search_dir.empty()) ? "" : (search_dir + "/")) << num_succeeds << "_" << num_fails << "_" << propose_count << ".json";
    return sstream.str();
}


namespace fdeep { namespace internal {

class cfv_mask_and_adjust_layer : public layer {
public:
    explicit cfv_mask_and_adjust_layer(const std::string& name) : layer(name) {}
protected:
    tensor5s apply_impl(const tensor5s& input) const override
    {
        const fdeep::tensor5& inp = input[0];
        const fdeep::tensor5& cfvs = input[1];

        const fdeep::float_type* inp_data = inp.as_vector()->data();
        const fdeep::float_type* cfvs_data = cfvs.as_vector()->data();
        bool mask[NUM_PLAYERS * NUM_VIEWPOINTS] = {0};
        for (int i = 0; i < NUM_ASSIGNMENTS; i++) {
            if (inp_data[i] > 0.0) {
                for (int player = 0; player < NUM_PLAYERS; player++) {
                    mask[NUM_VIEWPOINTS * player + ASSIGNMENT_TO_VIEWPOINT[i][player]] = true;
                }
            }   
        }
        int num_left = 0;
        for (int i = 0; i < NUM_PLAYERS * NUM_VIEWPOINTS; i++) {
            num_left += (int) (mask[i]);
        }

        fdeep::float_type masked_sum = 0.0;
        for (int i = 0; i < NUM_PLAYERS * NUM_VIEWPOINTS; i++) {
            if (mask[i]) {
                masked_sum += cfvs_data[i];
            }
        }
        fdeep::tensor5 result(cfvs.shape(), 0.0);
        fdeep::float_type* result_data = const_cast<fdeep::float_type*>(result.as_vector()->data());

        fdeep::float_type subtract_amount = masked_sum / num_left;
        for (int i = 0; i < NUM_PLAYERS * NUM_VIEWPOINTS; i++) {
            if (mask[i]) {
                result_data[i] = cfvs_data[i] - subtract_amount;
            }
        }

        return { result };
    }
};

inline layer_ptr create_cfv_mask_and_adjust_layer(
    const get_param_f&,
    const get_global_param_f&,
    const nlohmann::json&,
    const std::string& name) {
    return std::make_shared<cfv_mask_and_adjust_layer>(name);
}

const layer_creators custom_creator = { { "CFVMaskAndAdjustLayer", create_cfv_mask_and_adjust_layer } };

} } // namespace fdeep, namespace internal


std::shared_ptr<Model> load_model(const std::string& search_dir, const int num_succeeds, const int num_fails, const int propose_count) {
    auto cache_key = std::make_tuple(num_succeeds, num_fails, propose_count);
    if (model_cache.count(cache_key) != 0) {
        return model_cache[cache_key];
    }

    auto model_filename = get_model_filename(search_dir, num_succeeds, num_fails, propose_count);
    auto model = fdeep::load_model(
        model_filename,
        true,
        fdeep::cerr_logger,
        static_cast<fdeep::internal::float_type>(0.0001),
        fdeep::internal::custom_creator
    );
    auto model_ptr = std::make_shared<Model>(num_succeeds, num_fails, propose_count, std::move(model));

    model_cache[cache_key] = model_ptr;

    return model_ptr;
}

void Model::predict(const int proposer, const AssignmentProbs& input_probs, ViewpointVector* output_values) {
    fdeep::tensor5 input_tensor(fdeep::shape5(1, 1, 1, 1, NUM_ASSIGNMENTS + NUM_PLAYERS), 0.0);
    fdeep::float_type* input_data = const_cast<fdeep::float_type*>(input_tensor.as_vector()->data());
    input_data[proposer] = 1.0;
    for (int i = 0; i < NUM_ASSIGNMENTS; i++) {
        input_data[i + NUM_PLAYERS] = (fdeep::float_type) input_probs(i);
    }

    const auto result = this->model.predict({ input_tensor });

    const fdeep::float_type* output_data = result.front().as_vector()->data();

    for (int player = 0; player < NUM_PLAYERS; player++) {
        for (int viewpoint = 0; viewpoint < NUM_VIEWPOINTS; viewpoint++) {
            output_values[player](viewpoint) = (double) output_data[NUM_VIEWPOINTS*player + viewpoint];
        }
    }
}


void print_loaded_models(const std::string& search_dir) {
    if (model_cache.size() == 0) {
        std::cerr << "No models loaded." << std::endl;
        return;
    }
    for (const auto& pair : model_cache) {
        const auto& tuple = pair.first;
        std::cerr << get_model_filename(search_dir, std::get<0>(tuple), std::get<1>(tuple), std::get<2>(tuple)) << std::endl;
    }
}
