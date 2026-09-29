#pragma once

#include <string>
#include <vector>

// Load class names from a YOLO data.yaml file. Mirrors the format
// produced by Ultralytics: `names:` can be either a YAML sequence
// (block list or flow list) or a YAML map (id -> name).
//
// Throws std::runtime_error on:
//   - file open / parse failure
//   - missing `names:` key
//   - non-scalar entries in a sequence
//   - non-int keys in a map
//   - empty result
std::vector<std::string> load_class_names(const std::string& yaml_path);
