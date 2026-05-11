// wav_io.h
//
// Minimal WAV file reader restricted to 16 kHz mono 16-bit PCM, which
// is what Whisper and the diarizer expect.

#pragma once

#include <string>
#include <vector>


// Load a 16 kHz mono 16-bit PCM WAV file from `path` and return the
// samples as floats in [-1, 1]. Throws std::runtime_error on any
// format mismatch or read error.
std::vector<float> load_wav_16k_mono(const std::string& path);