// ring_buffer.h
//
// Single-producer single-consumer ring buffer for float audio samples.
// Producer (CoreAudio callback) writes; consumer (worker thread) reads.
// Overwrites the oldest sample when full.

#pragma once

#include <atomic>
#include <stdexcept>
#include <vector>


class RingBuffer {
private:
    std::vector<float> buf;
    std::atomic<int> head_{0};
    std::atomic<int> tail_{0};
    std::atomic<int> count_{0};
    int length = 0;

public:
    RingBuffer(int buffer_length);

    // Write a single sample. If the buffer is full, the oldest sample
    // is overwritten and the tail is advanced.
    void write(float value);

    // Read a single sample into `out`. Returns false if the buffer is
    // empty.
    bool read(float& out);

    // Read `batch_size` samples and append them to `out`. Returns false
    // if fewer than `batch_size` samples are available; in that case
    // nothing is read.
    bool batch_read(std::vector<float>& out, int batch_size);

    bool available() const {
        return count_ > 0;
    }

    int size() const {
        return count_;
    }
};