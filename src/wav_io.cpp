// wav_io.cpp
//
// Minimal WAV reader restricted to 16 kHz mono 16-bit PCM, which is
// what the rest of the pipeline expects. Supports arbitrary chunk
// ordering and unknown chunks (skipped).

#include "wav_io.h"

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>


namespace {

// Read exactly `size` bytes or throw. Plain ifstream::read silently
// short-reads at EOF; we want that to be a hard error here.
void read_exact(std::ifstream& file, char* data, const std::streamsize size,
                const std::string& path) {
    file.read(data, size);
    if (file.gcount() != size) {
        throw std::runtime_error("Unexpected end of WAV file: " + path);
    }
}

}  // namespace


std::vector<float> load_wav_16k_mono(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open WAV file: " + path);
    }

    // RIFF header: "RIFF" <size> "WAVE"
    char riff[4];
    read_exact(file, riff, 4, path);
    if (std::string(riff, 4) != "RIFF") {
        throw std::runtime_error("Not a RIFF file: " + path);
    }

    file.seekg(4, std::ios::cur);
    if (!file.good()) {
        throw std::runtime_error("Invalid RIFF header: " + path);
    }

    char wave[4];
    read_exact(file, wave, 4, path);
    if (std::string(wave, 4) != "WAVE") {
        throw std::runtime_error("Not a WAVE file: " + path);
    }

    // Walk chunks until both fmt and data have been seen. Unknown chunks
    // are skipped; odd-sized chunks have a trailing pad byte per RIFF spec.
    uint16_t audio_format = 0;
    uint16_t bits_per_sample = 0;
    uint16_t num_channels = 0;
    uint32_t sample_rate = 0;
    uint32_t data_size = 0;
    bool found_fmt = false;
    bool found_data = false;

    while (!found_data && file.good()) {
        char chunk_id[4];
        uint32_t chunk_size = 0;

        if (!file.read(chunk_id, 4)) {
            break;
        }
        if (!file.read(reinterpret_cast<char*>(&chunk_size), sizeof(chunk_size))) {
            break;
        }

        const std::string id(chunk_id, 4);
        if (id == "fmt ") {
            if (chunk_size < 16) {
                throw std::runtime_error("Invalid fmt chunk in WAV file: " + path);
            }

            read_exact(file, reinterpret_cast<char*>(&audio_format),
                       sizeof(audio_format), path);
            read_exact(file, reinterpret_cast<char*>(&num_channels),
                       sizeof(num_channels), path);
            read_exact(file, reinterpret_cast<char*>(&sample_rate),
                       sizeof(sample_rate), path);

            // Skip byte_rate (4) and block_align (2).
            file.seekg(6, std::ios::cur);
            if (!file.good()) {
                throw std::runtime_error("Invalid fmt chunk in WAV file: " + path);
            }

            read_exact(file, reinterpret_cast<char*>(&bits_per_sample),
                       sizeof(bits_per_sample), path);

            const auto bytes_consumed = static_cast<std::streamoff>(16);
            if (chunk_size > bytes_consumed) {
                file.seekg(static_cast<std::streamoff>(chunk_size) - bytes_consumed,
                           std::ios::cur);
                if (!file.good()) {
                    throw std::runtime_error("Invalid fmt chunk size in WAV file: " + path);
                }
            }

            if (chunk_size % 2 == 1) {
                file.seekg(1, std::ios::cur);
                if (!file.good()) {
                    throw std::runtime_error("Invalid fmt chunk padding in WAV file: " + path);
                }
            }

            found_fmt = true;
        } else if (id == "data") {
            data_size = chunk_size;
            found_data = true;
        } else {
            // Unknown chunk; skip past its body (and optional pad byte).
            file.seekg(static_cast<std::streamoff>(chunk_size), std::ios::cur);
            if (!file.good()) {
                throw std::runtime_error("Invalid WAV chunk size in file: " + path);
            }

            if (chunk_size % 2 == 1) {
                file.seekg(1, std::ios::cur);
                if (!file.good()) {
                    throw std::runtime_error("Invalid WAV chunk padding in file: " + path);
                }
            }
        }
    }

    // Validate format.
    if (!found_fmt) {
        throw std::runtime_error("No fmt chunk found: " + path);
    }
    if (!found_data) {
        throw std::runtime_error("No data chunk found: " + path);
    }
    if (audio_format != 1) {
        throw std::runtime_error("Only PCM WAV supported, got format " +
                                 std::to_string(audio_format) + ": " + path);
    }
    if (bits_per_sample != 16) {
        throw std::runtime_error("Only 16-bit WAV supported, got " +
                                 std::to_string(bits_per_sample) + "-bit: " + path);
    }
    if (num_channels != 1) {
        throw std::runtime_error("Only mono WAV supported, got " +
                                 std::to_string(num_channels) + " channels: " + path);
    }
    if (sample_rate != 16000) {
        throw std::runtime_error("Only 16 kHz WAV supported, got " +
                                 std::to_string(sample_rate) + " Hz: " + path);
    }
    if (data_size % sizeof(int16_t) != 0) {
        throw std::runtime_error("Invalid data chunk size for 16-bit WAV: " + path);
    }

    // Read PCM data and convert int16 -> float in [-1, 1].
    const size_t num_samples = data_size / sizeof(int16_t);
    std::vector<int16_t> raw(num_samples);
    read_exact(file, reinterpret_cast<char*>(raw.data()),
               static_cast<std::streamsize>(data_size), path);

    std::vector<float> audio(num_samples);
    for (size_t i = 0; i < num_samples; ++i) {
        audio[i] = static_cast<float>(raw[i]) / 32768.0f;
    }
    return audio;
}