/**
 * @file memory_pool.hpp
 * @brief Thread-safe buffer recycling for the Eigen math backend.
 * * This pool addresses the "Allocation Overhead" hinderance by preventing
 * frequent system calls to malloc/free during high-frequency sampling.
 */

#pragma once

#include <vector>
#include <unordered_map>
#include <mutex>
#include <memory>
#include <cstddef>

namespace isomorphism::math {

class MemoryPool {
public:
    /**
     * @brief Accesses the global singleton instance.
     * Thread-safe initialization is guaranteed by C++11 static rules.
     */
    static MemoryPool& instance() {
        static MemoryPool pool;
        return pool;
    }

    /**
     * @brief Acquires a float buffer of the specified size.
     * * If a buffer of this size exists in the cache, it is reused. Otherwise,
     * a new one is allocated. The returned shared_ptr uses a custom deleter
     * to return the pointer to the pool when its reference count hits zero.
     * * @param size Total number of float elements required.
     * @return A shared_ptr managing the recycled or new memory.
     */
    std::shared_ptr<float[]> acquire(size_t size) {
        if (size == 0) return nullptr;

        std::lock_guard<std::mutex> lock(mutex_);

        auto& bin = cache_[size];
        if (!bin.empty()) {
            float* ptr = bin.back();
            bin.pop_back();

            // Return shared_ptr with a lambda deleter that calls this->release()
            return std::shared_ptr<float[]>(ptr, [this, size](float* p) {
                this->release(p, size);
            });
        }

        // Cache miss: Allocate new memory from the heap
        return std::shared_ptr<float[]>(new float[size], [this, size](float* p) {
            this->release(p, size);
        });
    }

    /**
     * @brief Frees all cached memory currently held by the pool.
     */
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& [size, bin] : cache_) {
            for (float* ptr : bin) {
                delete[] ptr;
            }
            bin.clear();
        }
        cache_.clear();
    }

private:
    // Private constructor for Singleton pattern
    MemoryPool() = default;

    // Destructor ensures all managed raw pointers are deleted
    ~MemoryPool() {
        clear();
    }

    /**
     * @brief Internal method used by the custom deleter to recycle pointers.
     */
    void release(float* ptr, size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);
        cache_[size].push_back(ptr);
    }

    // Disable copying
    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;

    std::unordered_map<size_t, std::vector<float*>> cache_;
    std::mutex mutex_;
};

} // namespace isomorphism::math