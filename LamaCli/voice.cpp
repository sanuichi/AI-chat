#include "voice.h"
#include <Windows.h>
#include <pathcch.h>
#include <shlwapi.h>
#include <iostream>
#include <filesystem>


#define OPENJTALK_DICT_NAME L"open_jtalk_dic_utf_8-1.11"
#define MODEL_DIR_NAME L"models\\vvms"

voice::voice() {
}

voice::~voice() {
    if (synthesizer) {
        voicevox_synthesizer_delete(synthesizer);
    }
}

int voice::Init() {
    std::wcout << L"coreの初期化中" << std::endl;
    VoicevoxInitializeOptions initializeOptions = voicevox_make_default_initialize_options();
    std::string dict = GetOpenJTalkDict();

    auto load_ort_options = voicevox_make_default_load_onnxruntime_options();
    auto result = voicevox_onnxruntime_load_once(load_ort_options, &onnxruntime);
    if (result != VoicevoxResultCode::VOICEVOX_RESULT_OK) {
        OutErrorMessage(result);
        return -1;
    }

    OpenJtalkRc* open_jtalk;
    result = voicevox_open_jtalk_rc_new(dict.c_str(), &open_jtalk);
    if (result != VoicevoxResultCode::VOICEVOX_RESULT_OK) {
        OutErrorMessage(result);
        return -1;
    }

    result = voicevox_synthesizer_new(onnxruntime, open_jtalk, initializeOptions, &synthesizer);
    if (result != VoicevoxResultCode::VOICEVOX_RESULT_OK) {
        OutErrorMessage(result);
        voicevox_open_jtalk_rc_delete(open_jtalk);
        return -1;
    }
    voicevox_open_jtalk_rc_delete(open_jtalk);

    for (const auto& entry : std::filesystem::directory_iterator{ GetModelDir() }) {
        const auto path = entry.path();
        if (path.extension() != ".vvm") {
            continue;
        }
        VoicevoxVoiceModelFile* model;
        result = voicevox_voice_model_file_open(path.generic_u8string().c_str(), &model);
        if (result != VoicevoxResultCode::VOICEVOX_RESULT_OK) {
            OutErrorMessage(result);
            return -1;
        }
        result = voicevox_synthesizer_load_voice_model(synthesizer, model);
        if (result != VoicevoxResultCode::VOICEVOX_RESULT_OK) {
            OutErrorMessage(result);
            voicevox_voice_model_file_delete(model);
            return -1;
        }
        voicevox_voice_model_file_delete(model);
    }
    return 0;
}

int voice::VoicePlay(std::string data) {
    std::wcout << L"音声生成中" << std::endl;
    uintptr_t output_binary_size = 0;
    uint8_t* output_wav = nullptr;
    VoicevoxTtsOptions ttsOptions = voicevox_make_default_tts_options();

    auto result = voicevox_synthesizer_tts(synthesizer, data.c_str(), style_id, ttsOptions, &output_binary_size, &output_wav);

    if (result != VoicevoxResultCode::VOICEVOX_RESULT_OK) {
        OutErrorMessage(result);
        return -1;
    }

    std::wcout << L"音声再生中" << std::endl;
    PlaySound((LPCTSTR)output_wav, nullptr, SND_MEMORY);

    voicevox_wav_free(output_wav);
    return 0;
}

void voice::OutErrorMessage(VoicevoxResultCode messageCode) {
    const char* utf8Str = voicevox_error_result_to_message(messageCode);
    std::wstring wideStr = utf8_to_wide_cppapi(utf8Str);
    std::wcout << wideStr << std::endl;
}

std::string voice::GetOpenJTalkDict() {
    wchar_t buff[MAX_PATH] = { 0 };
    PathCchCombine(buff, MAX_PATH, GetExeDirectory().c_str(), OPENJTALK_DICT_NAME);
    std::string retVal = wide_to_utf8_cppapi(buff);
    return retVal;
}

std::string voice::GetModelDir() {
    wchar_t buff[MAX_PATH] = { 0 };
    PathCchCombine(buff, MAX_PATH, GetExeDirectory().c_str(), MODEL_DIR_NAME);
    std::string retVal = wide_to_utf8_cppapi(buff);
    return retVal;
}

std::wstring voice::GetExePath() {
    wchar_t buff[MAX_PATH] = { 0 };
    GetModuleFileNameW(nullptr, buff, MAX_PATH);
    return std::wstring(buff);
}

std::wstring voice::GetExeDirectory() {
    wchar_t buff[MAX_PATH] = { 0 };
    wcscpy_s(buff, MAX_PATH, GetExePath().c_str());
    PathRemoveFileSpecW(buff);
    return std::wstring(buff);
}

std::string voice::wide_to_utf8_cppapi(std::wstring const& src) {
    int num_chars = WideCharToMultiByte(CP_UTF8, 0, src.c_str(), -1, nullptr, 0, nullptr, nullptr);
    std::string dst(num_chars - 1, 0);
    WideCharToMultiByte(CP_UTF8, 0, src.c_str(), -1, &dst[0], num_chars, nullptr, nullptr);
    return dst;
}

std::wstring voice::utf8_to_wide_cppapi(std::string const& src) {
    int num_chars = MultiByteToWideChar(CP_UTF8, 0, src.c_str(), -1, nullptr, 0);
    std::wstring dst(num_chars - 1, 0);
    MultiByteToWideChar(CP_UTF8, 0, src.c_str(), -1, &dst[0], num_chars);
    return dst;
}
