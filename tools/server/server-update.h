#pragma once

#include <nlohmann/json.hpp>
#include <optional>
#include <regex>
#include <string>
#include <vector>
#include <thread>

namespace moeshift_update {
struct version {
    std::vector<std::string> core;
    int channel; // alpha=0, beta=1, stable=2
    std::string number;
};
inline int numeric_compare(const std::string & a, const std::string & b) {
    if (a.size() != b.size()) return a.size() < b.size() ? -1 : 1;
    return a == b ? 0 : (a < b ? -1 : 1);
}
inline std::optional<version> parse(const std::string & text) {
    static const std::regex pattern(
        R"(^v?(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:-(alpha|beta)(?:\.(0|[1-9][0-9]*))?)?(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$)");
    std::smatch m;
    if (!std::regex_match(text, m, pattern)) return {};
    return version{{m[1], m[2], m[3]}, m[4] == "alpha" ? 0 : (m[4] == "beta" ? 1 : 2), m[5]};
}
inline int compare(const version & a, const version & b) {
    for (size_t i = 0; i < 3; ++i) {
        const int n = numeric_compare(a.core[i], b.core[i]);
        if (n) return n;
    }
    if (a.channel != b.channel) return a.channel < b.channel ? -1 : 1;
    if (a.number.empty() != b.number.empty()) return a.number.empty() ? -1 : 1;
    return numeric_compare(a.number, b.number);
}
inline std::string select(const nlohmann::ordered_json & releases, const std::string & installed) {
    auto current = parse(installed);
    if (!current || !releases.is_array()) return {};
    auto best = *current;
    std::string result;
    for (const auto & release : releases) {
        if (!release.is_object() || !release.contains("draft") || !release["draft"].is_boolean() ||
            release["draft"].get<bool>() || !release.contains("tag_name") || !release["tag_name"].is_string() ||
            !release.contains("prerelease") || !release["prerelease"].is_boolean()) continue;
        const auto tag = release["tag_name"].get<std::string>();
        auto candidate = parse(tag);
        if (!candidate || release["prerelease"].get<bool>() != (candidate->channel != 2)) continue;
        if (candidate->channel >= current->channel && compare(*candidate, best) > 0) {
            best = *candidate;
            result = tag;
        }
    }
    return result;
}
// Only the public router owns this task; destruction joins before process shutdown.
class checker {
    std::thread task;
public:
    explicit checker(bool enabled);
    ~checker();
};
} // namespace moeshift_update
