#include "server-update.h"
#include "moeshift-version.h"
#include <cpp-httplib/httplib.h>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <unistd.h>

namespace moeshift_update {
namespace {
using json = nlohmann::ordered_json;
constexpr size_t max_body = 2 * 1024 * 1024;

std::filesystem::path cache_path() {
    const char * xdg = std::getenv("XDG_CACHE_HOME");
    const char * home = std::getenv("HOME");
    if (xdg && *xdg) return std::filesystem::path(xdg) / "moeshift/releases.json";
    if (home && *home) return std::filesystem::path(home) / "Library/Caches/moeshift/releases.json";
    return {};
}
void save(const std::filesystem::path & path, const json & value) {
    if (path.empty()) return;
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    if (ec) return;
    auto name = (path.parent_path() / "releases.XXXXXX").string();
    const int fd = mkstemp(name.data());
    if (fd < 0) return;
    close(fd);
    try {
        std::ofstream out(name);
        out << value.dump();
        out.close();
        if (!out) throw std::runtime_error("cache write failed");
        std::filesystem::rename(name, path);
    } catch (...) {
        std::filesystem::remove(name);
    }
}
void notify(const json & releases) {
    const auto tag = select(releases, MOESHIFT_VERSION);
    if (tag.empty()) return;
    // A GitHub release need not be available in the Homebrew tap yet.
    std::cerr << "moeshift: Update available: " << MOESHIFT_VERSION << " -> " << tag
              << "\n   Release and installation: https://github.com/davide-troiani/moeshift/releases/tag/" << tag << '\n';
}
std::optional<json> fetch(httplib::Client & client, std::chrono::milliseconds timeout = std::chrono::seconds(5)) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    json releases = json::array();
    for (int page = 1; ; ++page) {
        const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now());
        if (remaining.count() <= 0) return {};
        client.set_max_timeout(remaining);
        std::string body;
        const auto response = client.Get(
            "/repos/davide-troiani/moeshift/releases?per_page=100&page=" + std::to_string(page),
            {{"Accept", "application/vnd.github+json"}, {"User-Agent", "MoEShift/" MOESHIFT_VERSION}},
            [&](const char * data, size_t size) {
                if (body.size() + size > max_body || std::chrono::steady_clock::now() >= deadline) return false;
                body.append(data, size);
                return true;
            });
        if (!response || response->status != 200) return {};
        const auto batch = json::parse(body, nullptr, false);
        if (!batch.is_array()) return {};
        // Store only the fields needed for selection, never release bodies or assets.
        for (const auto & release : batch) {
            if (!release.is_object()) continue;
            releases.push_back({{"tag_name", release.value("tag_name", json())},
                                {"draft", release.value("draft", json())},
                                {"prerelease", release.value("prerelease", json())}});
        }
        if (releases.dump().size() > max_body) return {};
        if (batch.size() < 100) break;
    }
    return releases;
}
void check() {
    if (!parse(MOESHIFT_VERSION)) return;
    const auto path = cache_path();
    const auto now = std::time(nullptr);
    json cached;
    try {
        if (!path.empty() && std::filesystem::file_size(path) <= max_body) {
            std::ifstream in(path);
            in >> cached;
        }
    } catch (...) {}
    const bool has_cache = cached.is_object() && cached.contains("releases") && cached["releases"].is_array();
    if (has_cache) notify(cached["releases"]);
    if (has_cache && cached.contains("checked_at") && cached["checked_at"].is_number_integer()) {
        const auto checked = cached["checked_at"].get<long long>();
        if (checked <= now && now - checked < 86400) return;
    }
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    save(path, {{"checked_at", now}, {"releases", has_cache ? cached["releases"] : json::array()}});
    httplib::Client client("https://api.github.com");
    client.set_connection_timeout(2);
    client.set_read_timeout(2);
    client.set_write_timeout(2);
    const auto fetched = fetch(client);
    if (!fetched) return;
    const auto & releases = *fetched;
    if (!has_cache || select(cached["releases"], MOESHIFT_VERSION).empty()) notify(releases);
    save(path, {{"checked_at", now}, {"releases", releases}});
#endif
}
} // namespace
checker::checker(bool enabled) {
    try {
        if (enabled && parse(MOESHIFT_VERSION)) {
            task = std::thread([] { try { check(); } catch (...) { /* Updates never prevent inference. */ } });
        }
    } catch (...) { /* Thread allocation failure must not stop the server either. */ }
}
checker::~checker() { if (task.joinable()) task.join(); }
} // namespace moeshift_update
