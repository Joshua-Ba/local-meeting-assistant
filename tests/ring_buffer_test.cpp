#include "ring_buffer.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

TEST(RingBufferTest, StartsEmpty) {
    RingBuffer buffer(3);

    float value = 0.0f;
    EXPECT_FALSE(buffer.available());
    EXPECT_EQ(buffer.size(), 0);
    EXPECT_FALSE(buffer.read(value));
}

TEST(RingBufferTest, RejectsNonPositiveCapacity) {
    EXPECT_THROW(RingBuffer(0), std::invalid_argument);
    EXPECT_THROW(RingBuffer(-2), std::invalid_argument);
}

TEST(RingBufferTest, ReadsValuesInFifoOrder) {
    RingBuffer buffer(4);

    buffer.write(1.0f);
    buffer.write(2.0f);
    buffer.write(3.0f);

    EXPECT_TRUE(buffer.available());
    EXPECT_EQ(buffer.size(), 3);

    float value = 0.0f;
    ASSERT_TRUE(buffer.read(value));
    EXPECT_FLOAT_EQ(value, 1.0f);
    ASSERT_TRUE(buffer.read(value));
    EXPECT_FLOAT_EQ(value, 2.0f);
    ASSERT_TRUE(buffer.read(value));
    EXPECT_FLOAT_EQ(value, 3.0f);
    EXPECT_FALSE(buffer.read(value));
}

TEST(RingBufferTest, OverwritesOldestValueWhenFull) {
    RingBuffer buffer(3);

    buffer.write(1.0f);
    buffer.write(2.0f);
    buffer.write(3.0f);
    buffer.write(4.0f);

    EXPECT_EQ(buffer.size(), 3);

    float value = 0.0f;
    ASSERT_TRUE(buffer.read(value));
    EXPECT_FLOAT_EQ(value, 2.0f);
    ASSERT_TRUE(buffer.read(value));
    EXPECT_FLOAT_EQ(value, 3.0f);
    ASSERT_TRUE(buffer.read(value));
    EXPECT_FLOAT_EQ(value, 4.0f);
}

TEST(RingBufferTest, BatchReadRequiresEnoughDataAndAppendsOutput) {
    RingBuffer buffer(5);
    std::vector<float> out{99.0f};

    buffer.write(1.0f);
    buffer.write(2.0f);
    EXPECT_FALSE(buffer.batch_read(out, 3));
    EXPECT_EQ(out, std::vector<float>({99.0f}));
    EXPECT_EQ(buffer.size(), 2);

    buffer.write(3.0f);
    ASSERT_TRUE(buffer.batch_read(out, 3));
    EXPECT_EQ(out, std::vector<float>({99.0f, 1.0f, 2.0f, 3.0f}));
    EXPECT_EQ(buffer.size(), 0);
}

TEST(RingBufferTest, BatchReadRejectsNegativeSize) {
    RingBuffer buffer(3);
    std::vector<float> out;

    EXPECT_THROW(buffer.batch_read(out, -1), std::invalid_argument);
}
