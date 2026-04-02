#pragma once

#include <atomic>
#include <vector>

class RingBuffer
{
private:
    std::vector<float> buf;
    std::atomic<int> head_;
    std::atomic<int> tail_;
    int length = 0;

public:
    RingBuffer(int buffer_length);

    void write(float value);

    bool read(float& out);

    bool batch_read(std::vector<float>& out, int batch_size);

    bool available() {
        return head_ != tail_;
    }

    int size() const {
        return (head_ - tail_ + length) % length;
    }
};