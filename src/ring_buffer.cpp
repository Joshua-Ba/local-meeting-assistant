// ring_buffer.cpp
//
// Single-producer single-consumer ring buffer for float audio samples.
// Overwrites the oldest data when full.

#include "ring_buffer.h"


RingBuffer::RingBuffer(const int buffer_length) {
    if (buffer_length <= 0) {
        throw std::invalid_argument("RingBuffer length must be positive");
    }
    buf.assign(buffer_length, 0);
    length = buffer_length;
}


void RingBuffer::write(const float value) {
    const int head = head_.load();
    buf[head] = value;
    head_ = (head + 1) % length;

    // When the buffer is full, advance the tail to drop the oldest sample.
    if (count_ == length) {
        tail_ = (tail_.load() + 1) % length;
    } else {
        ++count_;
    }
}


bool RingBuffer::read(float& out) {
    if (!available()) return false;
    const int tail = tail_.load();
    out = buf[tail];
    tail_ = (tail + 1) % length;
    --count_;
    return true;
}


bool RingBuffer::batch_read(std::vector<float>& out, int batch_size) {
    if (batch_size < 0) {
        throw std::invalid_argument("batch_size must be non-negative");
    }
    if (size() < batch_size) {
        return false;
    }

    const int tail = tail_.load();
    for (int i = 0; i < batch_size; ++i) {
        out.push_back(buf[(tail + i) % length]);
    }
    tail_ = (tail + batch_size) % length;
    count_ -= batch_size;
    return true;
}