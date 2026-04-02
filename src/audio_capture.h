#pragma once
#include <string>
#include <CoreAudio/CoreAudio.h>
#include "ring_buffer.h"
#include <iostream>

struct AudioCallbackData {
    RingBuffer* buffer;
    int resample_factor;
};

class AudioCapture
{
private:
    RingBuffer* buf;
    std::string audio_device_name;
    AudioDeviceIOProcID procID = nullptr;
    AudioDeviceID blackholeDeviceID = 0;
    AudioCallbackData callbackData;
    int resample_factor;

public:
    AudioCapture(RingBuffer* buffer_name, const std::string& device_name, const int resample_factor);

    bool start();

    void stop();

};