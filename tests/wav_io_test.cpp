#include "wav_io.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {
void append_fourcc(std::vector<uint8_t>& bytes, const char* fourcc) {
    for (int i = 0; i < 4; ++i) {
        bytes.push_back(static_cast<uint8_t>(fourcc[i]));
    }
}

void append_u16_le(std::vector<uint8_t>& bytes, const uint16_t value) {
    bytes.push_back(static_cast<uint8_t>(value & 0xFFu));
    bytes.push_back(static_cast<uint8_t>((value >> 8) & 0xFFu));
}

void append_u32_le(std::vector<uint8_t>& bytes, const uint32_t value) {
    bytes.push_back(static_cast<uint8_t>(value & 0xFFu));
    bytes.push_back(static_cast<uint8_t>((value >> 8) & 0xFFu));
    bytes.push_back(static_cast<uint8_t>((value >> 16) & 0xFFu));
    bytes.push_back(static_cast<uint8_t>((value >> 24) & 0xFFu));
}

std::filesystem::path write_test_wav(const std::string& filename,
                                     const uint16_t num_channels,
                                     const uint32_t sample_rate,
                                     const uint16_t bits_per_sample,
                                     const std::vector<int16_t>& samples,
                                     const bool include_list_chunk) {
    std::vector<uint8_t> bytes;
    append_fourcc(bytes, "RIFF");
    append_u32_le(bytes, 0);  // patched later
    append_fourcc(bytes, "WAVE");

    append_fourcc(bytes, "fmt ");
    append_u32_le(bytes, 16);
    append_u16_le(bytes, 1);  // PCM
    append_u16_le(bytes, num_channels);
    append_u32_le(bytes, sample_rate);
    const uint16_t block_align = static_cast<uint16_t>(num_channels * (bits_per_sample / 8));
    const uint32_t byte_rate = sample_rate * block_align;
    append_u32_le(bytes, byte_rate);
    append_u16_le(bytes, block_align);
    append_u16_le(bytes, bits_per_sample);

    if (include_list_chunk) {
        const std::vector<uint8_t> list_payload = {
            'I', 'N', 'F', 'O',
            'I', 'S', 'F', 'T',
            'L', 'a', 'v', 'f',
            '6', '1', '.', '7',
            '.', '1', '0', '0',
            0,   0,   0,   0,
        };
        append_fourcc(bytes, "LIST");
        append_u32_le(bytes, static_cast<uint32_t>(list_payload.size()));
        bytes.insert(bytes.end(), list_payload.begin(), list_payload.end());
    }

    append_fourcc(bytes, "data");
    append_u32_le(bytes, static_cast<uint32_t>(samples.size() * sizeof(int16_t)));
    for (const int16_t sample : samples) {
        append_u16_le(bytes, static_cast<uint16_t>(sample));
    }

    const uint32_t riff_size = static_cast<uint32_t>(bytes.size() - 8);
    bytes[4] = static_cast<uint8_t>(riff_size & 0xFFu);
    bytes[5] = static_cast<uint8_t>((riff_size >> 8) & 0xFFu);
    bytes[6] = static_cast<uint8_t>((riff_size >> 16) & 0xFFu);
    bytes[7] = static_cast<uint8_t>((riff_size >> 24) & 0xFFu);

    const auto path = std::filesystem::current_path() / filename;
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    out.close();
    return path;
}
}

TEST(WavIoTest, LoadWav16kMonoSkipsListChunkAndReadsActualSamples) {
    const std::vector<int16_t> samples = {1000, -1000, 2000, -2000};
    const auto path = write_test_wav("wav_io_with_list.wav", 1, 16000, 16, samples, true);

    const auto audio = load_wav_16k_mono(path.string());

    ASSERT_EQ(audio.size(), samples.size());
    for (size_t i = 0; i < samples.size(); ++i) {
        EXPECT_FLOAT_EQ(audio[i], static_cast<float>(samples[i]) / 32768.0f);
    }
}

TEST(WavIoTest, LoadWav16kMonoRejectsUnsupportedSampleRate) {
    const auto path = write_test_wav("wav_io_bad_rate.wav", 1, 8000, 16, {1000, -1000}, false);

    try {
        (void)load_wav_16k_mono(path.string());
        FAIL() << "Expected load_wav_16k_mono to throw";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find("16 kHz"), std::string::npos);
    }
}
