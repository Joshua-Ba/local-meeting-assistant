#include "audio_capture.h"


OSStatus audioCallback(AudioDeviceID device,
                       const AudioTimeStamp* now,
                       const AudioBufferList* inputData,
                       const AudioTimeStamp* inputTime,
                       AudioBufferList* outputData,
                       const AudioTimeStamp* outputTime,
                       void* clientData) {

    auto* data = static_cast<AudioCallbackData*>(clientData);
    auto* samples = static_cast<float*>(inputData->mBuffers[0].mData);
    int num_floats = inputData->mBuffers[0].mDataByteSize / sizeof(float);

    // Resample from BlackHole's native format (e.g. 48kHz stereo) to 16kHz mono
    // by taking every Nth sample, where N = (source_rate / target_rate) * channels
    for (int i=0; i<num_floats; i++) {
        if (i % data->resample_factor == 0) {
            data->buffer->write(samples[i]);
        }
    }
    return noErr;
}


AudioCapture::AudioCapture(RingBuffer* buffer_name, const std::string& device_name, const int resample_factor) : buf(buffer_name), audio_device_name(device_name), resample_factor(resample_factor){}

bool AudioCapture::start() {
    AudioObjectPropertyAddress propertyAddress;
    propertyAddress.mSelector = kAudioHardwarePropertyDevices;
    propertyAddress.mScope = kAudioObjectPropertyScopeGlobal;
    propertyAddress.mElement = kAudioObjectPropertyElementMain;

    UInt32 dataSize = 0;
    AudioObjectGetPropertyDataSize(kAudioObjectSystemObject, &propertyAddress, 0, nullptr, &dataSize);
    auto deviceCount = dataSize / sizeof(AudioDeviceID);

    std::cout << std::to_string(deviceCount) << std::endl;

    std::vector<AudioDeviceID> devices(deviceCount);
    AudioObjectGetPropertyData(kAudioObjectSystemObject, &propertyAddress, 0, nullptr, &dataSize, devices.data());

    AudioObjectPropertyAddress propertyAddress2;
    propertyAddress2.mSelector = kAudioDevicePropertyDeviceNameCFString;
    propertyAddress2.mScope = kAudioObjectPropertyScopeGlobal;
    propertyAddress2.mElement = kAudioObjectPropertyElementMain;


    for (const auto& device : devices) {
        CFStringRef deviceName = nullptr;
        UInt32 nameSize = sizeof(CFStringRef);
        AudioObjectGetPropertyData(device, &propertyAddress2, 0, nullptr, &nameSize, &deviceName);
        if (deviceName) {
            char name[256];
            CFStringGetCString(deviceName, name, sizeof(name), kCFStringEncodingUTF8);
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
    AudioDeviceCreateIOProcID(blackholeDeviceID, audioCallback, &callbackData, &procID);
    AudioDeviceStart(blackholeDeviceID, procID);
    return true;
}

void AudioCapture::stop() {
    AudioDeviceStop(blackholeDeviceID, procID);
    AudioDeviceDestroyIOProcID(blackholeDeviceID, procID);
    procID = nullptr;
    blackholeDeviceID = 0;
}
