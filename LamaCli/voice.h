#pragma once

#include <string>
#include "voicevox_core.h"

class voice {
public:
    voice();
    ~voice();
    int Init();
    int VoicePlay(std::string data);

private:
    VoicevoxSynthesizer* synthesizer = nullptr;
    const VoicevoxOnnxruntime* onnxruntime = nullptr;


    void OutErrorMessage(VoicevoxResultCode messageCode);
    std::string GetOpenJTalkDict();
    std::string GetModelDir();
    std::wstring GetExePath();
    std::wstring GetExeDirectory();
    std::string wide_to_utf8_cppapi(std::wstring const& src);
    std::wstring utf8_to_wide_cppapi(std::string const& src);

public:
    int32_t style_id = 8;

};
