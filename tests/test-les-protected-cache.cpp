#include "../src/llama-moe-protected-cache.h"
#include <iostream>
#include <random>
#include <set>

int main() {
    les_protected_cache cache;
    cache.configure(15, 38, 512);
    std::vector<int32_t> ids(38, -1), slots(512, -1);
    std::vector<uint64_t> ages(38, 0);
    std::mt19937 random(42);
    uint64_t age = 0;
    auto install = [&](int expert, const std::vector<uint8_t> & requested) {
        int slot = slots[expert];
        if (slot >= 0) {
            cache.hit(expert, slot, ids, ages, requested, {});
        } else {
            slot = cache.miss(expert, ids, ages, requested, {});
            assert(slot >= 0);
            assert(ids[slot] != -2);
            assert(ids[slot] < 0 || requested.empty() || !requested[ids[slot]]);
            if (ids[slot] >= 0) slots[ids[slot]] = -1;
            ids[slot] = expert;
            slots[expert] = slot;
        }
        ages[slot] = ++age;
        return slot;
    };
    for (int step = 0; step < 3000; ++step) {
        cache.begin_step();
        std::set<int> used;
        while (used.size() < 10) used.insert(random() % 100);
        std::vector<uint8_t> requested(512, 0);
        for (int expert : used) requested[expert] = 1;
        for (int expert : used) install(expert, requested);
        for (int expert : used) assert(slots[expert] >= 0 && ids[slots[expert]] == expert);

        // A speculative writer owns this slot until publication or cancellation.
        const int pending = install(100 + random() % 400, {});
        slots[ids[pending]] = -1;
        ids[pending] = -2;
        install(511, {});
        assert(ids[pending] == -2);
        ids[pending] = -1;
        cache.reconcile(ids);
        assert(cache.count(1) <= 15 && cache.count(2) <= 23);
        for (size_t i = 0; i < ids.size(); ++i) {
            if (ids[i] >= 0) assert(slots[ids[i]] == int(i));
            else assert(!cache.group[i]);
        }
        if (step % 257 == 0) {
            std::fill(ids.begin(), ids.end(), -1);
            std::fill(slots.begin(), slots.end(), -1);
            cache.reset(ids);
            assert(!cache.count(1) && !cache.count(2) && cache.steps == 0);
        }
    }
    cache.frequency[0] = 8;
    cache.steps = 128;
    cache.begin_step();
    assert(cache.frequency[0] == 4 && cache.steps == 129);
    std::cout << "Protected cache invariants passed (3000 steps)\n";
}
