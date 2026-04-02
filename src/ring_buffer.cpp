#include "ring_buffer.h"

RingBuffer::RingBuffer(const int buffer_length) : buf(buffer_length, 0), length(buffer_length) {}

void RingBuffer::write(const float value){
    buf[head_] = value;
    head_ = (head_ + 1) % length;
}

bool RingBuffer::read(float& out) {
    if (!available()) return false;
    out = buf[tail_];
    tail_ = (tail_ + 1) % length;
    return true;
}

bool RingBuffer::batch_read(std::vector<float>& out, int batch_size ) {
    if (size() >= batch_size) {
        for (int i = 0; i<batch_size;i++) {
            out.push_back(buf[(tail_+i)%length]);
        }
        tail_ = (tail_ + batch_size) % length;
        return true;
    }
    return false;
}