#define MOESHIFT_VERSION "0.1.0-alpha.1"
#include "../tools/server/server-update.cpp"
#include <cassert>
#include <atomic>
#include <sstream>

using namespace moeshift_update;
using json = nlohmann::ordered_json;
static json release(const std::string & tag, bool draft = false) {
    return {{"tag_name", tag}, {"draft", draft}, {"prerelease", tag.find('-') != std::string::npos}};
}
int main() {
    assert(!parse("01.2.3"));
    assert(!parse("1.2.3-alpha.01"));
    assert(!parse("1.2.3-rc.1"));
    assert(!parse("1.2.3-dev"));
    assert(parse("v1.2.3+build.7"));
    assert(compare(*parse("1.2.3+foo"), *parse("1.2.3+bar")) == 0);
    assert(compare(*parse("0.1.0-alpha.9"), *parse("0.1.0-alpha.10")) < 0);
    assert(compare(*parse("0.1.0-alpha"), *parse("0.1.0-alpha.0")) < 0);
    assert(compare(*parse("1.9.0"), *parse("1.10.0")) < 0);
    json releases = {release("0.3.0-alpha.1"), release("0.2.0-beta.2"), release("0.1.1"), release("9.0.0", true)};
    assert(select(releases, "0.1.0") == "0.1.1");
    assert(select(releases, "0.1.0-alpha.1") == "0.3.0-alpha.1");
    assert(select(releases, "0.1.0-beta.1") == "0.2.0-beta.2");
    assert(select(releases, "0.4.0").empty());
    assert(select(releases, "dev").empty());
    assert(select(json::array({release("0.1.0")}), "0.1.0-beta.1") == "0.1.0");
    assert(select(json::array({release("0.1.0-alpha.1")}), "0.1.0-beta.1").empty());
    releases.push_back({{"tag_name", "99.0.0"}, {"draft", false}, {"prerelease", true}});
    releases.push_back({{"tag_name", 42}});
    assert(select(releases, "0.1.0") == "0.1.1");

    httplib::Server server;
    int requests = 0;
    std::atomic<int> mode{0};
    server.Get("/repos/davide-troiani/moeshift/releases", [&](const httplib::Request & req, httplib::Response & res) {
        ++requests;
        if (mode == 1) { res.status = 500; return; }
        if (mode == 2) { res.set_content("invalid json", "application/json"); return; }
        if (mode == 3) { res.set_content(std::string(max_body + 1, 'x'), "application/json"); return; }
        if (mode == 4) { std::this_thread::sleep_for(std::chrono::milliseconds(300)); }

        if (req.get_param_value("page") == "1") {
            json batch = json::array();
            for (int i = 0; i < 100; ++i) batch.push_back(release("0.1.0"));
            res.set_content(batch.dump(), "application/json");
        } else res.set_content(json::array({release("0.1.1")}).dump(), "application/json");
    });
    const int port = server.bind_to_any_port("127.0.0.1");
    assert(port > 0);
    std::thread thread([&] { server.listen_after_bind(); });
    server.wait_until_ready();
    httplib::Client client("127.0.0.1", port);
    auto fetched = fetch(client);
    assert(fetched && requests == 2 && select(*fetched, "0.1.0") == "0.1.1");
    assert(!fetch(client, std::chrono::milliseconds(0)));
    for (int error_mode : {1, 2, 3}) {
        mode = error_mode;
        assert(!fetch(client));
    }
    mode = 4;
    const auto start = std::chrono::steady_clock::now();
    assert(!fetch(client, std::chrono::milliseconds(50)));
    assert(std::chrono::steady_clock::now() - start < std::chrono::milliseconds(250));
    server.stop();
    thread.join();
    assert(!fetch(client, std::chrono::milliseconds(100)));

    char temp[] = "/tmp/moeshift-update-test.XXXXXX";
    assert(mkdtemp(temp));
    setenv("XDG_CACHE_HOME", temp, 1);
    const auto path = cache_path();
    save(path, {{"checked_at", std::time(nullptr)}, {"releases", json::array({release("0.1.0-beta.1")})}});
    std::ostringstream output;
    auto previous = std::cerr.rdbuf(output.rdbuf());
    { checker disabled(false); }
    assert(output.str().empty());
    { checker enabled(true); }
    std::cerr.rdbuf(previous);
    assert(output.str().find("0.1.0-beta.1") != std::string::npos);
    assert(output.str().find("Update available") == output.str().rfind("Update available"));
    save(path, {{"checked_at", std::time(nullptr)}, {"releases", json::array({release("0.1.0")})}});
    assert(select(json::array({release("0.2.0-alpha.1")}), "0.1.0").empty());
    { std::ofstream broken(path); broken << "broken"; }
#ifndef CPPHTTPLIB_OPENSSL_SUPPORT
    { checker enabled(true); }
#endif
    std::filesystem::remove_all(temp);
}
