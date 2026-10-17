#include "yaml_classes.h"

#include <map>
#include <stdexcept>
#include <string>

#include <yaml-cpp/yaml.h>

std::vector<std::string> load_class_names(const std::string& yaml_path) {
    YAML::Node root;
    try {
        root = YAML::LoadFile(yaml_path);
    }
    catch (const YAML::Exception& e) {
        throw std::runtime_error(
            std::string("Failed to parse data.yaml at ") + yaml_path + ": "
            + e.what()
        );
    }

    if (!root["names"]) {
        throw std::runtime_error(
            "data.yaml at " + yaml_path
            + " is missing the required 'names:' key."
        );
    }

    YAML::Node names = root["names"];
    std::vector<std::string> classes;

    if (names.IsSequence()) {
        // Format A: - person / - bicycle / ...   (or flow list)
        for (const auto& node : names) {
            if (!node.IsScalar()) {
                throw std::runtime_error(
                    "data.yaml 'names:' list contains a non-scalar entry."
                );
            }
            classes.push_back(node.as<std::string>());
        }
    }
    else if (names.IsMap()) {
        // Format B: { 0: person, 1: bicycle, ... }
        // Sort by integer key to preserve class-id ordering.
        std::map<int, std::string> by_id;
        for (auto it = names.begin(); it != names.end(); ++it) {
            int id;
            try {
                id = it->first.as<int>();
            }
            catch (const YAML::Exception&) {
                throw std::runtime_error(
                    "data.yaml 'names:' map has non-integer key: '"
                    + it->first.as<std::string>() + "'"
                );
            }
            by_id[id] = it->second.as<std::string>();
        }
        for (const auto& kv : by_id) {
            classes.push_back(kv.second);
        }
    }
    else {
        throw std::runtime_error(
            "data.yaml 'names:' must be a sequence or a map."
        );
    }

    if (classes.empty()) {
        throw std::runtime_error(
            "data.yaml 'names:' has no entries (empty list/map)."
        );
    }

    return classes;
}
