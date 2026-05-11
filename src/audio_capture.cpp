// audio_capture.cpp
//
// macOS CoreAudio capture from a named device (e.g. BlackHole) into a
// RingBuffer. Resamples by simple decimation: every N-th float is kept,
// where N covers both sample-rate downsampling and channel downmixing.

#include "audio_capture.h"


// =====================================================================
// CoreAudio callback
// =====================================================================

OSStatus audioCallback(AudioDeviceID device,
                       const AudioTimeStamp* now,
                       const AudioBufferList* inputData,
                       const AudioTimeStamp* inputTime,
                       AudioBufferList* outputData,
                       const AudioTimeStamp* outputTime,
                       void* clientData) {
    auto* data = static_cast<AudioCallbackData*>(clientData);
    if (!data || !data->buffer || data->resample_factor <= 0 ||
        !inputData || inputData->mNumberBuffers == 0) {
        return noErr;
    }

    auto* samples = static_cast<float*>(inputData->mBuffers[0].mData);
    if (!samples) {
        return noErr;
    }

    const int num_floats = inputData->mBuffers[0].mDataByteSize / sizeof(float);

    // Resample from BlackHole's native format (e.g. 48kHz stereo) to 16kHz
    // mono by taking every Nth sample, where N = (source_rate / target_rate)
    // * channels.
    for (int i = 0; i < num_floats; ++i) {
        if (i % data->resample_factor == 0) {
            data->buffer->write(samples[i]);
        }
    }
    return noErr;
}


// =====================================================================
// AudioCapture
// =====================================================================

AudioCapture::AudioCapture(RingBuffer* buffer_name,
                           const std::string& device_name,
                           const int resample_factor)
    : buf(buffer_name),
      audio_device_name(device_name),
      resample_factor(resample_factor) {}


bool AudioCapture::start() {
    AudioObjectPropertyAddress propertyAddress;
    propertyAddress.mSelector = kAudioHardwarePropertyDevices;
    propertyAddress.mScope = kAudioObjectPropertyScopeGlobal;
    propertyAddress.mElement = kAudioObjectPropertyElementMain;

    UInt32 dataSize = 0;
    AudioObjectGetPropertyDataSize(kAudioObjectSystemObject, &propertyAddress,
                                   0, nullptr, &dataSize);
    const auto deviceCount = dataSize / sizeof(AudioDeviceID);

    std::cout << std::to_string(deviceCount) << std::endl;

    std::vector<AudioDeviceID> devices(deviceCount);
    AudioObjectGetPropertyData(kAudioObjectSystemObject, &propertyAddress,
                               0, nullptr, &dataSize, devices.data());

    AudioObjectPropertyAddress propertyAddress2;
    propertyAddress2.mSelector = kAudioDevicePropertyDeviceNameCFString;
    propertyAddress2.mScope = kAudioObjectPropertyScopeGlobal;
    propertyAddress2.mElement = kAudioObjectPropertyElementMain;

    // Scan devices for a name that contains `audio_device_name` (substring match).
    for (const auto& device : devices) {
        CFStringRef deviceName = nullptr;
        UInt32 nameSize = sizeof(CFStringRef);
        AudioObjectGetPropertyData(device, &propertyAddress2, 0, nullptr,
                                   &nameSize, &deviceName);
        if (deviceName) {
            char name[256];
            CFStringGetCString(deviceName, name, sizeof(name),
                               kCFStringEncodingUTF8);
            if (std::string(name).find(audio_device_name) != std::string::npos) {
                std::cout << name << std::endl;
                blackholeDeviceID = device;
            }
        }
    }

    if (blackholeDeviceID == 0) {
        std::cerr << "device not found" << std::endl;
        return false;
    }

    callbackData = {buf, resample_factor};
    AudioDeviceCreateIOProcID(blackholeDeviceID, audioCallback,
                              &callbackData, &procID);
    AudioDeviceStart(blackholeDeviceID, procID);
    return true;
}


void AudioCapture::stop() {
    AudioDeviceStop(blackholeDeviceID, procID);
    AudioDeviceDestroyIOProcID(blackholeDeviceID, procID);
    procID = nullptr;
    blackholeDeviceID = 0;
}