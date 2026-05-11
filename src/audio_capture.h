// audio_capture.h
//
// macOS CoreAudio capture into a RingBuffer. The capture is driven by an
// AudioDeviceIOProc callback which writes resampled mono floats into the
// buffer. Resampling is done by simple decimation: keep every N-th sample.

#pragma once

#include <CoreAudio/CoreAudio.h>
#include <iostream>
#include <string>

#include "ring_buffer.h"


// Context passed to the CoreAudio callback. Lives in the AudioCapture
// instance for the duration of capture.
struct AudioCallbackData {
    RingBuffer* buffer;
    int resample_factor;
};


// CoreAudio IOProc callback. Reads from the device's input buffer,
// decimates by `resample_factor`, and writes the result to `buffer`.
OSStatus audioCallback(AudioDeviceID device,
                       const AudioTimeStamp* now,
                       const AudioBufferList* inputData,
                       const AudioTimeStamp* inputTime,
                       AudioBufferList* outputData,
                       const AudioTimeStamp* outputTime,
                       void* clientData);


class AudioCapture {
private:
    RingBuffer* buf;
    std::string audio_device_name;
    AudioDeviceIOProcID procID = nullptr;
    AudioDeviceID blackholeDeviceID = 0;
    AudioCallbackData callbackData;
    int resample_factor;

public:
    AudioCapture(RingBuffer* buffer_name,
                 const std::string& device_name,
                 const int resample_factor);

    // Find the named device and start the IOProc. Returns false if no
    // device matched `device_name` (substring match).
    bool start();

    // Stop the IOProc and release the device handle.
    void stop();
};