#include "audio_capture.h"

#include <gtest/gtest.h>

TEST(AudioCaptureTest, CallbackResamplesByWritingEveryNthFloat) {
    RingBuffer buffer(8);
    AudioCallbackData callback_data{&buffer, 2};
    float samples[] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    AudioBufferList input{};
    input.mNumberBuffers = 1;
    input.mBuffers[0].mNumberChannels = 1;
    input.mBuffers[0].mDataByteSize = sizeof(samples);
    input.mBuffers[0].mData = samples;

    EXPECT_EQ(audioCallback(0, nullptr, &input, nullptr, nullptr, nullptr, &callback_data), noErr);

    EXPECT_EQ(buffer.size(), 3);
    float value = 0.0f;
    ASSERT_TRUE(buffer.read(value));
    EXPECT_FLOAT_EQ(value, 0.0f);
    ASSERT_TRUE(buffer.read(value));
    EXPECT_FLOAT_EQ(value, 2.0f);
    ASSERT_TRUE(buffer.read(value));
    EXPECT_FLOAT_EQ(value, 4.0f);
}

TEST(AudioCaptureTest, CallbackIgnoresInvalidInputSafely) {
    RingBuffer buffer(4);
    AudioCallbackData invalid_factor{&buffer, 0};

    EXPECT_EQ(audioCallback(0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr), noErr);
    EXPECT_EQ(audioCallback(0, nullptr, nullptr, nullptr, nullptr, nullptr, &invalid_factor), noErr);
    EXPECT_EQ(buffer.size(), 0);
}
