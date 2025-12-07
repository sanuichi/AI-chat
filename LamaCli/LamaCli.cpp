#include "arg.h"
#include "common.h"
#include "console.h"
#include "log.h"
#include "sampling.h"
#include "llama.h"
#include "chat.h"
#include "voice.h"

#define NOMINMAX
#include <windows.h>

#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <windows.h>
#include <mmdeviceapi.h>
#include <audioclient.h>
#include <iomanip>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <cmath>
#include <algorithm>
#include <locale>
#include <codecvt>
#include <samplerate.h>

#include <io.h>
#include <fcntl.h>

#include <Eigen/Dense>
#include <sndfile.h>
#include <fftw3.h>
#include <onnxruntime_cxx_api.h>
#include <portaudio.h>

#include <cuda_runtime.h>
#include <cudnn.h>

#include <conio.h>

#include "whisper.h"
#include "SpeakerIdentifier.h"

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined (_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <signal.h>
#endif

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267)
#endif

#pragma comment(lib, "ole32.lib")




// WAVファイルヘッダー構造体
struct WAVHeader {
    char riff[4] = { 'R', 'I', 'F', 'F' };
    uint32_t fileSize;
    char wave[4] = { 'W', 'A', 'V', 'E' };
    char fmt[4] = { 'f', 'm', 't', ' ' };
    uint32_t fmtSize = 16;
    uint16_t audioFormat = 1; // PCM
    uint16_t numChannels;
    uint32_t sampleRate;
    uint32_t byteRate;
    uint16_t blockAlign;
    uint16_t bitsPerSample;
    char data[4] = { 'd', 'a', 't', 'a' };
    uint32_t dataSize;
};

static llama_context** g_ctx;
static llama_model** g_model;
static common_sampler** g_smpl;
static common_params* g_params;
static std::vector<llama_token>* g_input_tokens;
static std::ostringstream* g_output_ss;
static std::vector<llama_token>* g_output_tokens;
static bool is_interacting = false;
static bool need_insert_eot = false;
static voice g_voice;

const char* MODEL_PATH = "D:/work/models/ggml-large-v3.bin";

const int TARGET_RATE = 16000;
const double CHUNK_SECONDS = 0.5;
const double VAD_RMS_THRESHOLD = 0.005;
const int SILENCE_LIMIT = 2;
const int ADAPTIVE_WINDOW = 30;     // 20 → 30 に拡大


std::mutex qmutex;
std::condition_variable qcv;
std::queue<std::vector<float>> audio_queue;
std::atomic<bool> stop_flag(false);

std::queue<std::vector<std::string>> speak_queue;
std::mutex smutex;
std::condition_variable scv;


std::string AssistantName = "春日部つむぎ";

using namespace SpeakerID;

// ============================================
// エンコーディング処理の統一
// ============================================

void InitializeConsole() {
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    _setmode(_fileno(stdout), _O_BINARY);
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stderr), _O_BINARY);
}

std::string utf8_to_sjis(const std::string& utf8str) {
    // UTF-8 → UTF-16（ワイド文字列）へ変換
    int wlen = MultiByteToWideChar(CP_UTF8, 0, utf8str.c_str(), -1, nullptr, 0);
    std::wstring wstr(wlen, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, utf8str.c_str(), -1, &wstr[0], wlen);

    // UTF-16 → Shift_JIS（CP932）へ変換
    int slen = WideCharToMultiByte(CP_ACP, 0, wstr.c_str(), -1, nullptr, 0, nullptr, nullptr);
    std::string sjisstr(slen, '\0');
    WideCharToMultiByte(CP_ACP, 0, wstr.c_str(), -1, &sjisstr[0], slen, nullptr, nullptr);

    return sjisstr;
}

void WriteUTF8(const std::string& utf8_text) {
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD fileType = GetFileType(hConsole);

    if (fileType == FILE_TYPE_CHAR) {
        int wlen = MultiByteToWideChar(CP_UTF8, 0, utf8_text.c_str(), -1, NULL, 0);
        if (wlen > 0) {
            std::vector<wchar_t> wbuf(wlen);
            MultiByteToWideChar(CP_UTF8, 0, utf8_text.c_str(), -1, wbuf.data(), wlen);
            DWORD written;
            WriteConsoleW(hConsole, wbuf.data(), static_cast<DWORD>(wcslen(wbuf.data())), &written, NULL);
        }
    }
    else {
        DWORD written;
        WriteFile(hConsole, utf8_text.c_str(), static_cast<DWORD>(utf8_text.length()), &written, NULL);
    }
}

std::string ReadUTF8Input() {
    HANDLE hConsole = GetStdHandle(STD_INPUT_HANDLE);

    // ハンドルの有効性チェック
    if (hConsole == INVALID_HANDLE_VALUE || hConsole == NULL) {
        std::cerr << "Error: Invalid console handle" << std::endl;
        return "";
    }

    // 現在のコンソールモードを保存
    DWORD originalMode = 0;
    if (!GetConsoleMode(hConsole, &originalMode)) {
        std::cerr << "Error: Failed to get console mode" << std::endl;
        return "";
    }

    // ライン入力モードを有効化
    DWORD mode = ENABLE_LINE_INPUT | ENABLE_ECHO_INPUT | ENABLE_PROCESSED_INPUT;
    if (!SetConsoleMode(hConsole, mode)) {
        std::cerr << "Error: Failed to set console mode" << std::endl;
        return "";
    }

    std::wstring wbuf;
    wbuf.resize(4096);  // バッファを事前確保
    DWORD read = 0;
    std::string result;

    // UTF-16でコンソールから読み取り
    BOOL readResult = ReadConsoleW(hConsole, &wbuf[0], 4095, &read, NULL);

    // 元のコンソールモードを復元
    SetConsoleMode(hConsole, originalMode);

    if (!readResult) {
        DWORD error = GetLastError();
        std::cerr << "Error: ReadConsoleW failed with error code: " << error << std::endl;
        return "";
    }

    if (read > 0) {
        // 末尾の改行を削除
        while (read > 0 && (wbuf[read - 1] == L'\n' || wbuf[read - 1] == L'\r')) {
            read--;
        }

        if (read > 0) {
            // UTF-16 → UTF-8変換
            int utf8_len = WideCharToMultiByte(
                CP_UTF8,
                0,
                wbuf.data(),
                static_cast<int>(read),
                NULL,
                0,
                NULL,
                NULL
            );

            if (utf8_len > 0) {
                std::vector<char> utf8_buf(utf8_len + 1);
                WideCharToMultiByte(
                    CP_UTF8,
                    0,
                    wbuf.data(),
                    static_cast<int>(read),
                    utf8_buf.data(),
                    utf8_len,
                    NULL,
                    NULL
                );
                utf8_buf[utf8_len] = '\0';
                result = std::string(utf8_buf.data());
            }
        }
    }

    return result;
}
std::string ReadUTF8Input2() {
    std::string line;
    if (std::getline(std::cin, line)) {
        // 改行コードを削除
        if (!line.empty() && (line.back() == '\r' || line.back() == '\n')) {
            line.pop_back();
        }
        if (!line.empty() && (line.back() == '\r' || line.back() == '\n')) {
            line.pop_back();
        }
    }
    return line;
}

class MicrophoneRecorder {
public:
    MicrophoneRecorder(int sampleRate = 16000, int channels = 1)
        : sampleRate_(sampleRate), channels_(channels), stream_(nullptr), isRecording_(false) {
        if (Pa_Initialize() != paNoError) {
            throw std::runtime_error("PortAudio initialization error");
        }
    }

    ~MicrophoneRecorder() {
        stop();
        Pa_Terminate();
    }

    bool start() {
        PaStreamParameters inputParams;
        inputParams.device = Pa_GetDefaultInputDevice();
        if (inputParams.device == paNoDevice) {
            std::cerr << "No input device found" << std::endl;
            return false;
        }
        inputParams.channelCount = channels_;
        inputParams.sampleFormat = paFloat32;
        inputParams.suggestedLatency = Pa_GetDeviceInfo(inputParams.device)->defaultLowInputLatency;
        inputParams.hostApiSpecificStreamInfo = nullptr;

        recordedData_.clear();
        isRecording_ = true;

        if (Pa_OpenStream(&stream_, &inputParams, nullptr, sampleRate_, 256, paClipOff,
            recordCallback, this) != paNoError) {
            return false;
        }
        if (Pa_StartStream(stream_) != paNoError) {
            return false;
        }
        return true;
    }

    void stop() {
        if (stream_) {
            isRecording_ = false;
            Pa_StopStream(stream_);
            Pa_CloseStream(stream_);
            stream_ = nullptr;
        }
    }

    bool isRecording() const { return isRecording_; }
    std::vector<float> getRecordedData() const { return recordedData_; }

    double getRecordedDuration() const {
        return static_cast<double>(recordedData_.size()) / sampleRate_;
    }

private:
    int sampleRate_, channels_;
    PaStream* stream_;
    std::atomic<bool> isRecording_;
    std::vector<float> recordedData_;

    static int recordCallback(const void* input, void*, unsigned long frameCount,
        const PaStreamCallbackTimeInfo*, PaStreamCallbackFlags, void* userData) {
        auto* recorder = static_cast<MicrophoneRecorder*>(userData);
        if (recorder->isRecording_ && input) {
            const float* in = static_cast<const float*>(input);
            recorder->recordedData_.insert(recorder->recordedData_.end(),
                in, in + frameCount * recorder->channels_);
        }
        return paContinue;
    }
};

std::vector<float> loadWavFile(const std::string& path, int targetSampleRate = 16000) {
    SF_INFO sfInfo;
    SNDFILE* file = sf_open(path.c_str(), SFM_READ, &sfInfo);
    if (!file) {
        std::cerr << "Error: Cannot open " << path << std::endl;
        return {};
    }
    if (sfInfo.samplerate != targetSampleRate) {
        std::cerr << "Warning: Sample rate is not " << targetSampleRate << "Hz ("
            << sfInfo.samplerate << "Hz)" << std::endl;
    }

    std::vector<float> audio(static_cast<size_t>(sfInfo.frames) * static_cast<size_t>(sfInfo.channels));
    sf_read_float(file, audio.data(), audio.size());
    sf_close(file);

    if (sfInfo.channels > 1) {
        std::vector<float> mono(static_cast<size_t>(sfInfo.frames));
        for (sf_count_t i = 0; i < sfInfo.frames; ++i) {
            float sum = 0.0f;
            for (int c = 0; c < sfInfo.channels; ++c)
                sum += audio[static_cast<size_t>(i) * sfInfo.channels + c];
            mono[static_cast<size_t>(i)] = sum / sfInfo.channels;
        }
        return mono;
    }
    return audio;
}

class SpeakerSelectSystem {
public:
    SpeakerSelectSystem(const std::string& modelDir, const std::string& onnxPath, const Config& config)
        : identifier_(modelDir, onnxPath, config), config_(config) {
        std::cout << "Speaker identification system initialized" << std::endl;
    }

    bool enrollSpeakerFromFile(const std::string& speakerName, const std::string& wavPath) {
        auto audio = loadWavFile(wavPath, config_.sampleRate);
        if (audio.empty()) {
            std::cerr << "Failed to load audio: " << wavPath << std::endl;
            return false;
        }
        return identifier_.enrollSpeaker(speakerName, audio);
    }

    bool enrollSpeaker(const std::string& speakerName, const std::vector<float>& audio) {
        if (audio.empty()) {
            std::cerr << "Audio data is empty" << std::endl;
            return false;
        }
        return identifier_.enrollSpeaker(speakerName, audio);
    }

    bool enrollSpeakerFromMic(const std::string& speakerName, double durationSec = 3.0) {
        try {
            MicrophoneRecorder recorder(config_.sampleRate);
            if (!recorder.start()) {
                std::cerr << "Failed to start recording" << std::endl;
                return false;
            }

            std::cout << "Recording... (" << durationSec << " seconds)" << std::endl;
            Pa_Sleep(static_cast<long>(durationSec * 1000));
            recorder.stop();

            auto audio = recorder.getRecordedData();
            std::cout << "Recording completed: " << recorder.getRecordedDuration() << " seconds" << std::endl;

            return identifier_.enrollSpeaker(speakerName, audio);
        }
        catch (const std::exception& e) {
            std::cerr << "Microphone error: " << e.what() << std::endl;
            return false;
        }
    }

    IdentificationResult identifyFromFile(const std::string& wavPath) {
        auto audio = loadWavFile(wavPath, config_.sampleRate);
        if (audio.empty()) {
            std::cerr << "Failed to load audio: " << wavPath << std::endl;
            return IdentificationResult();
        }
        return identifier_.identify(audio);
    }

    IdentificationResult identify(const std::vector<float>& audio) {
        if (audio.empty()) {
            std::cerr << "Audio data is empty" << std::endl;
            return IdentificationResult();
        }
        return identifier_.identify(audio);
    }

    struct RecordingResult {
        std::vector<float> audio;
        double duration;
        bool success;
    };

    RecordingResult recordFromMic() {
        RecordingResult result{ {}, 0.0, false };

        try {
            MicrophoneRecorder recorder(config_.sampleRate);
            if (!recorder.start()) {
                std::cerr << "Failed to start recording" << std::endl;
                return result;
            }

            std::cout << "\nRecording... (Press Enter to stop)" << std::endl;
            std::cin.get();

            recorder.stop();
            result.audio = recorder.getRecordedData();
            result.duration = recorder.getRecordedDuration();
            result.success = true;

            std::cout << "Recording completed: " << std::fixed << std::setprecision(2)
                << result.duration << " seconds (" << result.audio.size() << " samples)" << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "Microphone error: " << e.what() << std::endl;
        }

        return result;
    }

    IdentificationResult identifyFromMic() {
        auto recording = recordFromMic();

        if (!recording.success || recording.audio.empty()) {
            return IdentificationResult();
        }

        if (recording.duration < 0.5) {
            std::cout << "\nWarning: Recording duration is too short (minimum 0.5 seconds recommended)" << std::endl;
        }

        return identifier_.identify(recording.audio);
    }

    bool removeSpeaker(const std::string& speakerName) {
        return identifier_.removeSpeaker(speakerName);
    }

    std::vector<std::string> getSpeakerNames() const {
        return identifier_.getSpeakerNames();
    }

    size_t getSpeakerCount() const {
        return identifier_.getSpeakerCount();
    }

    void printResult(const IdentificationResult& result) const {
        std::cout << "\n================================" << std::endl;
        std::cout << "       Result" << std::endl;
        std::cout << "================================" << std::endl;

        if (result.isUnknown) {
            std::cout << "Speaker: Unknown" << std::endl;
        }
        else {
            std::cout << "Speaker: " << result.speakerName << std::endl;
        }

        std::cout << "Confidence: " << std::fixed << std::setprecision(2)
            << (result.confidence * 100.0) << "%" << std::endl;

        std::cout << "\n[Confidence Level] " << result.getConfidenceDescription() << std::endl;

        std::cout << "\n[All Speaker Rankings]" << std::endl;
        for (size_t i = 0; i < result.allScores.size(); ++i) {
            const auto& [name, score] = result.allScores[i];
            std::cout << "  " << (i + 1) << ". " << name << ": "
                << std::fixed << std::setprecision(2) << (score * 100.0) << "%" << std::endl;
        }
        std::cout << "================================\n" << std::endl;
    }

    void printSystemInfo() const {
        std::cout << "\n[System Information]" << std::endl;
        std::cout << "Sample rate: " << config_.sampleRate << " Hz" << std::endl;
        std::cout << "Mel frequency bands: " << config_.nMels << " dimensions" << std::endl;
        std::cout << "Registered speakers: " << getSpeakerCount() << " people" << std::endl;
    }

private:
    SpeakerIdentifier identifier_;
    Config config_;
};

double rms(const float* data, size_t n) {
    double s = 0.0;
    for (size_t i = 0; i < n; ++i) s += double(data[i]) * double(data[i]);
    return n ? std::sqrt(s / n) : 0.0;
}

double zcr(const float* data, size_t n) {
    int zero_crossings = 0;
    for (size_t i = 1; i < n; ++i) {
        if ((data[i - 1] >= 0 && data[i] < 0) ||
            (data[i - 1] < 0 && data[i] >= 0)) {
            zero_crossings++;
        }
    }
    return n ? double(zero_crossings) / double(n) : 0.0;
}

double update_adaptive_rms_threshold2(double current_rms,
    std::vector<double>& recent_rms,
    size_t window_size = 20,
    double scale = 1.5)
{
    recent_rms.push_back(current_rms);
    if (recent_rms.size() > window_size) recent_rms.erase(recent_rms.begin());

    double avg = 0.0;
    for (double r : recent_rms) avg += r;
    avg /= recent_rms.size();

    return avg * scale;
}

// ZCR範囲の調整
bool is_speech(const float* data, size_t n, double rms_threshold) {
    double r = rms(data, n);
    double z = zcr(data, n);

    const double ZCR_MIN = 0.05;  // 0.03 → 0.05 に引き上げ（足音などの低周波ノイズ除去）
    const double ZCR_MAX = 0.6;   // 0.6 (変更なし)

    // RMSとZCRの両方が条件を満たす必要がある
    bool rms_ok = r > rms_threshold;
    bool zcr_ok = z > ZCR_MIN && z < ZCR_MAX;

    // デバッグ出力（必要に応じて）
    // std::cout << "RMS: " << r << " ZCR: " << z << " Speech: " << (rms_ok && zcr_ok) << std::endl;

    return rms_ok && zcr_ok;
}

// 適応的しきい値更新の改善
double update_adaptive_rms_threshold(double current_rms,
    std::vector<double>& recent_rms,
    size_t window_size = 30,    // 20 → 30
    double scale = 2.0)         // 1.5 → 2.0
{
    recent_rms.push_back(current_rms);
    if (recent_rms.size() > window_size) recent_rms.erase(recent_rms.begin());

    // 平均値ではなく中央値を使う（外れ値に強い）
    std::vector<double> sorted = recent_rms;
    std::sort(sorted.begin(), sorted.end());

    double median;
    size_t n = sorted.size();
    if (n % 2 == 0) {
        median = (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
    }
    else {
        median = sorted[n / 2];
    }

    // 最小しきい値を設定
    const double MIN_THRESHOLD = 0.02; // 0.015 → 0.02 に引き上げ（最小音量底上げ）
    return std::max(MIN_THRESHOLD, median * scale);
}

bool is_speech2(const float* data, size_t n, double rms_threshold) {
    double r = rms(data, n);
    double z = zcr(data, n);

    const double ZCR_MIN = 0.05;
    const double ZCR_MAX = 0.6;

    return (r > rms_threshold && z > ZCR_MIN && z < ZCR_MAX);
}

std::vector<float> resample_float_mono(const std::vector<float>& in, int src_rate, int dst_rate) {
    if (src_rate == dst_rate) return in;
    double ratio = double(dst_rate) / double(src_rate);
    size_t est_out = size_t(double(in.size()) * ratio) + 10;
    std::vector<float> out(est_out);

    SRC_DATA src_data{};
    src_data.data_in = in.data();
    src_data.input_frames = static_cast<long>(in.size());
    src_data.data_out = out.data();
    src_data.output_frames = static_cast<long>(out.size());
    src_data.src_ratio = ratio;
    src_data.end_of_input = 1;

    int err = src_simple(&src_data, SRC_SINC_MEDIUM_QUALITY, 1);
    if (err != 0) {
        std::cerr << "libsamplerate error: " << src_strerror(err) << std::endl;
        return {};
    }
    out.resize(src_data.output_frames_gen);
    return out;
}

bool SaveWavFile2(const char* filename, std::vector<float> data)
{
    // パラメータのチェック
    if (data.empty()) {
        return false;
    }

    // WAVファイルのデフォルト設定
    const uint16_t channels = 1;        // モノラル
    const uint32_t sampleRate = 16000;  // 44.1kHz
    const uint16_t bitsPerSample = 16;  // 16bit

    // ファイルを開く
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    // float データを 16bit PCM に変換
    std::vector<int16_t> pcmData;
    pcmData.reserve(data.size());

    for (float sample : data) {
        // floatの値を -1.0 ~ 1.0 の範囲にクランプ
        sample = std::max(-1.0f, std::min(1.0f, sample));

        // 16bit PCM の範囲 (-32768 ~ 32767) に変換
        int16_t pcmSample = static_cast<int16_t>(sample * 32767.0f);
        pcmData.push_back(pcmSample);
    }

    // WAVヘッダーの設定
    WAVHeader header;
    header.numChannels = channels;
    header.sampleRate = sampleRate;
    header.bitsPerSample = bitsPerSample;
    header.byteRate = sampleRate * channels * (bitsPerSample / 8);
    header.blockAlign = channels * (bitsPerSample / 8);
    header.dataSize = static_cast<uint32_t>(pcmData.size() * sizeof(int16_t));
    header.fileSize = sizeof(WAVHeader) - 8 + header.dataSize;

    // ヘッダーを書き込み
    file.write(reinterpret_cast<const char*>(&header), sizeof(header));
    if (file.fail()) {
        file.close();
        return false;
    }

    // PCMデータを書き込み
    file.write(reinterpret_cast<const char*>(pcmData.data()),
        pcmData.size() * sizeof(int16_t));
    if (file.fail()) {
        file.close();
        return false;
    }

    file.close();
    return true;
}



// ============================================
// ハイパスフィルター (150Hz以下をカット)
// ============================================
class HighPassFilter {
public:
    HighPassFilter(float cutoffFreq, float sampleRate)
        : prevInput(0.0f), prevOutput(0.0f) {
        float dt = 1.0f / sampleRate;
        float rc = 1.0f / (2.0f * 3.14159265359f * cutoffFreq);
        alpha = rc / (rc + dt);
    }

    void process(std::vector<float>& data) {
        for (float& sample : data) {
            float input = sample;
            float output = alpha * (prevOutput + input - prevInput);
            prevInput = input;
            prevOutput = output;
            sample = output;
        }
    }

private:
    float alpha;
    float prevInput;
    float prevOutput;
};


// ============================================
// ローパスフィルター (4000Hz以上をカット)
// ============================================
class LowPassFilter {
public:
    LowPassFilter(float cutoffFreq, float sampleRate)
        : prevOutput(0.0f) {
        float dt = 1.0f / sampleRate;
        float rc = 1.0f / (2.0f * 3.14159265359f * cutoffFreq);
        alpha = dt / (rc + dt);
    }

    void process(std::vector<float>& data) {
        for (float& sample : data) {
            float input = sample;
            float output = prevOutput + alpha * (input - prevOutput);
            prevOutput = output;
            sample = output;
        }
    }

private:
    float alpha;
    float prevOutput;
};

void Enrollment_thread()
{
    int silence_count = 0;
    int active_count = 0;
    std::vector<float> accumulated_chunk;
    std::vector<double> recent_rms;

    Config config;
    config.sampleRate = 16000;
    config.nMels = 80;
    config.cudaDeviceId = 0;
    config.useFP16 = false;
    config.useCPUFallback = true;
    config.minConfidence = 0.55;

    std::string onnxModelPath = "speaker_resnet.onnx";
    SpeakerSelectSystem system("speaker_models", onnxModelPath, config);

    // ハイパスフィルターの初期化 (150Hz)
    HighPassFilter hpf(150.0f, 16000.0f);
    // ローパスフィルターの初期化 (4000Hz)
    LowPassFilter lpf(4000.0f, 16000.0f);

    while (!stop_flag) {
        WriteUTF8("\n--- 新しい話者を登録 ---\n");
        WriteUTF8("話者名を入力 (終了: 'quit'): ");

        std::string speakerName = ReadUTF8Input();

        if (speakerName == "quit" || speakerName == "q") {
            break;
        }
        if (speakerName.empty()) {
            WriteUTF8("⚠ 話者名を入力してください\n");
            continue;
        }

        WriteUTF8("VOICEVOXを使いますか  0:使わない 1:使います");
        std::string character = ReadUTF8Input();

        // 状態をリセット
        silence_count = 0;
        active_count = 0;
        bool recording_started = false;
        int pre_speech_silence = 0;  // 録音開始前の連続無音カウント
        accumulated_chunk.clear();
        recent_rms.clear();

        int chara  = std::stoi(character);

        if (chara == 1) {

            std::string speakID = ReadUTF8Input();

            if (speakID.empty()) {
                WriteUTF8("⚠ キャラクターNoを入力してください\n");
                continue;
            }

            g_voice.style_id = std::stoi(speakID);


            WriteUTF8("しゃべる文章を入力: ");
            std::string speakString = ReadUTF8Input();

            if (speakString.empty()) {
                WriteUTF8("⚠ 文章を入力してください\n");
                continue;
            }

            WriteUTF8("四国めたん\tあまあま\t0\n");
            WriteUTF8("ずんだもん\tノーマル\t3\n");
            WriteUTF8("春日部つむぎ\tノーマル\t8\n");
            WriteUTF8("九州そら\tノーマル\t16\n");
            WriteUTF8("もち子さん\tノーマル\t20\n");
            WriteUTF8("\nキャラクターNoを入力: ");


            // 音声キューとバッファをクリア
            {
                std::lock_guard<std::mutex> lk(qmutex);
                while (!audio_queue.empty()) {
                    audio_queue.pop();
                }
            }


            if (!speakString.empty()) {
                WriteUTF8("\n音声を再生します...\n");
                g_voice.VoicePlay(speakString);

                // 音声再生開始を待つ
                Sleep(500);
            }
        }
        else {

            // 音声キューとバッファをクリア
            {
                std::lock_guard<std::mutex> lk(qmutex);
                while (!audio_queue.empty()) {
                    audio_queue.pop();
                }
            }

            WriteUTF8("\n音声を待機中...(話し始めると自動で録音開始します)\n");

            WriteUTF8("マイクに向かって喋ってください");
        }



        // 録音用の無音判定閾値
        // ※ 0.5秒 × 4 = 2.0秒の無音で録音終了
        // ※ 文中の短い間（Silence=1〜2）は継続、文末の長い無音で終了
        const int RECORDING_SILENCE_LIMIT = 4;


        while (!stop_flag) {

            std::unique_lock<std::mutex> lk(qmutex);
            qcv.wait(lk, [] { return !audio_queue.empty() || stop_flag.load(); });
            if (stop_flag && audio_queue.empty()) break;

            auto chunk = std::move(audio_queue.front());
            audio_queue.pop();
            lk.unlock();

            if (chunk.empty()) continue;

            // ハイパスフィルター適用 (足音対策)
            hpf.process(chunk);
            // ローパスフィルター適用 (拍手・食器音対策)
            lpf.process(chunk);


            double current_rms = rms(chunk.data(), chunk.size());
            double adaptive_threshold = update_adaptive_rms_threshold(current_rms, recent_rms, 20, 2.5);

            // 録音中は閾値を下げる（文末の小さい音も拾う）
            double speech_threshold = recording_started ? adaptive_threshold * 0.7 : adaptive_threshold;

            bool speech = is_speech(chunk.data(), chunk.size(), speech_threshold);

            // 音声検出時の処理
            if (speech) {
                if (!recording_started) {
                    recording_started = true;
                    pre_speech_silence = 0;
                    WriteUTF8("🎤 録音開始！\n");
                }
                silence_count = 0;
                active_count++;
            }
            else {
                // 無音検出
                if (recording_started && active_count > 0) {
                    // 録音開始後の無音をカウント
                    silence_count++;
                }
                else if (!recording_started) {
                    // 録音開始前の無音カウント（タイムアウト用）
                    pre_speech_silence++;
                }
            }

            // 録音開始後のみデータを蓄積
            if (recording_started) {
                accumulated_chunk.insert(accumulated_chunk.end(), chunk.begin(), chunk.end());
            }

            // タイムアウトチェック（30秒音声なし）
            if (!recording_started && pre_speech_silence > 60) {  // 0.5秒 × 60 = 30秒
                WriteUTF8("⚠ タイムアウト: 音声が検出されませんでした\n");
                break;
            }

            if (recording_started) {
                std::cout << "RMS=" << std::fixed << std::setprecision(4) << current_rms
                    << "  Threshold=" << speech_threshold
                    << "  Speech=" << speech
                    << "  Active=" << active_count
                    << "  Silence=" << silence_count
                    << "  Duration=" << std::setprecision(1)
                    << (accumulated_chunk.size() / (float)TARGET_RATE) << "s"
                    << std::endl;
            }

            // 音声終了判定（無音が3秒続いたら終了）
            const int RECORDING_SILENCE_LIMIT = 6;  // 0.5秒 × 6 = 3秒
            if (silence_count >= RECORDING_SILENCE_LIMIT && active_count > 0 && recording_started) {
                WriteUTF8("\n🛑 録音終了（無音検出）\n");

                if (accumulated_chunk.size() >= TARGET_RATE * 0.5) {

                    // 末尾の無音を削除（2.5秒分 = RECORDING_SILENCE_LIMIT - 1チャンク分の余裕）
                    int silence_samples = static_cast<int>(TARGET_RATE * 2.5); // 2.5秒
                    int trim_size = std::min(silence_samples, static_cast<int>(accumulated_chunk.size()) - TARGET_RATE / 2);

                    if (trim_size > 0) {
                        accumulated_chunk.resize(accumulated_chunk.size() - trim_size);
                        std::cout << "末尾の無音を削除: " << (trim_size / (float)TARGET_RATE)
                            << "秒 (残り: " << (accumulated_chunk.size() / (float)TARGET_RATE) << "秒)" << std::endl;
                    }

                    // UTF-8 → Shift-JIS変換してファイル名作成
                    std::string filename_utf8 = speakerName + ".wav";
                    std::string filename_sjis = utf8_to_sjis(filename_utf8);

                    // Shift-JISのファイル名でWAVファイル保存
                    bool saveResult = SaveWavFile2(filename_sjis.c_str(), accumulated_chunk);

                    if (saveResult) {
                        WriteUTF8("✓ 音声ファイル保存成功\n");

                        // 話者登録（UTF-8の名前を使用）
                        WriteUTF8("話者登録中...\n");
                        bool enrollResult = system.enrollSpeaker(speakerName, accumulated_chunk);

                        if (enrollResult) {
                            WriteUTF8("✓ 話者登録成功\n");
                        }
                        else {
                            WriteUTF8("✗ 話者登録失敗\n");
                        }
                    }
                    else {
                        WriteUTF8("✗ 音声ファイル保存失敗\n");
                    }
                }
                else {
                    WriteUTF8("⚠ 音声が短すぎます（最低0.5秒必要）\n");
                }

                // 次の登録に備えてリセット
                accumulated_chunk.clear();
                silence_count = 0;
                active_count = 0;
                break;
            }
        }
    }

    WriteUTF8("\n登録モードを終了します\n");
}

// ファイルの先頭付近に追加
std::vector<float> trimTrailingSilence(const std::vector<float>& audio,
                                    int sampleRate,
                                    double silenceDuration,
                                    double keepMargin = 0.2) 
{

    if (audio.empty()) return audio;

    int trim_samples = static_cast<int>(sampleRate * (silenceDuration - keepMargin));
    int min_keep = sampleRate / 2; // 最低0.5秒

    int actual_trim = std::min(
        trim_samples,
        static_cast<int>(audio.size()) - min_keep
    );

    if (actual_trim > 0) {
        std::vector<float> trimmed(audio.begin(), audio.end() - actual_trim);
        std::cout << "✂️ 末尾無音削除: " << std::fixed << std::setprecision(2)
            << (actual_trim / (float)sampleRate) << "秒 "
            << "(残り: " << (trimmed.size() / (float)sampleRate) << "秒)"
            << std::endl;
        return trimmed;
    }

    return audio;
}


void worker_thread_func() {
    whisper_context_params cparams = whisper_context_default_params();
    struct whisper_context* ctx = whisper_init_from_file_with_params(MODEL_PATH, cparams);

    if (!ctx) {
        std::cerr << "whisper_init_from_file failed\n";
        return;
    }

    struct whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.print_progress = false;
    params.print_realtime = false;
    params.print_timestamps = false;
    params.language = "auto";

    int silence_count = 0;
    int active_count = 0;
    std::vector<float> accumulated_chunk;
    std::vector<float> prev_chunk;
    std::vector<double> recent_rms;

    Config config;
    config.sampleRate = 16000;
    config.nMels = 80;
    config.cudaDeviceId = 0;
    config.useFP16 = false;
    config.useCPUFallback = true;
    config.minConfidence = 0.55;

    std::string onnxModelPath = "speaker_resnet.onnx";
    SpeakerSelectSystem system("speaker_models", onnxModelPath, config);

    // ハイパスフィルターの初期化 (150Hz)
    HighPassFilter hpf(150.0f, 16000.0f);
    // ローパスフィルターの初期化 (4000Hz)
    LowPassFilter lpf(4000.0f, 16000.0f);

    system.printSystemInfo();

    int spekercount;
    if ((spekercount = system.getSpeakerCount()) == 0) {
        std::cerr << "\nWarning: No registered speakers" << std::endl;
        std::cerr << "   Please register speakers in registration mode first" << std::endl;
    }

    WriteUTF8("\n[Registered Speakers]\n");
    auto speakers = system.getSpeakerNames();

    for (size_t i = 0; i < speakers.size(); ++i) {
        std::string text = "  " + std::to_string(i + 1) + ". " + speakers[i] + "\n";
        WriteUTF8(text);
    }

    bool recording_started = false;
    int pre_speech_silence = 0;  // 録音開始前の連続無音カウント
    int savecnt = 0;

    std::unique_lock<std::mutex> lk(qmutex);
    qcv.wait(lk, [] { return !audio_queue.empty() || stop_flag.load(); });

    while (!audio_queue.empty()) {
        audio_queue.pop();
    }
    lk.unlock();

    while (!stop_flag) {

        // 音声データ
        std::unique_lock<std::mutex> lk(qmutex);
        qcv.wait(lk, [] { return !audio_queue.empty() || stop_flag.load(); });
        if (stop_flag && audio_queue.empty()) break;

        auto chunk = std::move(audio_queue.front());
        audio_queue.pop();
        lk.unlock();

        if (chunk.empty()) continue;

        // ハイパスフィルター適用 (足音対策)
        hpf.process(chunk);
        // ローパスフィルター適用 (拍手・食器音対策)
        lpf.process(chunk);


        double current_rms = rms(chunk.data(), chunk.size());
        double adaptive_threshold = update_adaptive_rms_threshold(current_rms, recent_rms, 30, 2.5); // 1.5 -> 2.5, 20 -> 30 (登録モードと統一)

        // 録音中は閾値を下げる（文末の小さい音も拾う）
        double speech_threshold = recording_started ? adaptive_threshold * 0.7 : adaptive_threshold;

        bool speech = is_speech(chunk.data(), chunk.size(), speech_threshold);

        // 音声検出時の処理
        if (speech) {
            if (!recording_started) {
                recording_started = true;
                pre_speech_silence = 0;
                if (!prev_chunk.empty()) {
                    // 前回の無音チャンクの後半半分だけを追加
                    size_t offset = prev_chunk.size() / 2;
                    accumulated_chunk.insert(accumulated_chunk.end(), prev_chunk.begin() + offset, prev_chunk.end());
                }
                WriteUTF8("🎤 録音開始！\n");
            }
            silence_count = 0;
            active_count++;
        }
        else {
            // 無音検出
            if (recording_started && active_count > 0) {
                // 録音開始後の無音をカウント
                silence_count++;
            }
            else if (!recording_started) {
                // 録音開始前の無音カウント（タイムアウト用）
                pre_speech_silence++;
                prev_chunk = chunk;
            }
        }

        // 録音開始後のみデータを蓄積
        if (recording_started) {
            accumulated_chunk.insert(accumulated_chunk.end(), chunk.begin(), chunk.end());
        }


        if (recording_started) {

            std::cout << "RMS=" << std::fixed << std::setprecision(4) << current_rms
                << "  Threshold=" << speech_threshold
                << "  Speech=" << speech
                << "  Active=" << active_count
                << "  Silence=" << silence_count
                << std::endl;
        }


        if (!speech) {


            // silence_count が SILENCE_LIMIT以上で、蓄積チャンクがあり、active_countが0より大きい場合に処理を行う    
            if (silence_count >= SILENCE_LIMIT && !accumulated_chunk.empty() && active_count > 0) {

                // チャット処理
                // 話者識別　音声からテキスト変換　LLMへの受け渡しなど

                std::vector<float> to_process = accumulated_chunk;
                accumulated_chunk.clear();
                silence_count = 0;
                active_count = 0;


                if (to_process.size() >= TARGET_RATE * 0.5) {
                    to_process = trimTrailingSilence(
                        to_process,
                        TARGET_RATE,
                        SILENCE_LIMIT * 0.5,  // 1秒
                        0.2  // 0.2秒残す（Whisper用）
                    );

                    ///////////////////////////////////////////////////////////
                    // UTF-8 → Shift-JIS変換してファイル名作成
                    std::string filename_utf8 = std::to_string(savecnt++) + ".wav";
                    std::string filename_sjis = utf8_to_sjis(filename_utf8);

                    // Shift-JISのファイル名でWAVファイル保存
                    bool saveResult = SaveWavFile2(filename_sjis.c_str(), to_process);


					if (savecnt >= 10) savecnt = 0;

                    ///////////////////////////////////////////////////////////

                    auto result = system.identify(to_process);

                    if (!result.isUnknown) {
                        std::string speaker_info = "\n[話者: " + result.speakerName +
                            " (信頼度: " + std::to_string(int(result.confidence * 100)) + "%)] \n";
                        WriteUTF8(speaker_info);
                    }
                    else {
                        std::string speaker_info = "\n[話者: UNKNOWN (信頼度: " +
                            std::to_string(int(result.confidence * 100)) + "%)] ";
                        WriteUTF8(speaker_info);
                    }
                
                    if (AssistantName != result.speakerName) {
                        int rc = whisper_full(ctx, params, to_process.data(), (int)to_process.size());
                        if (rc == 0) {
                            const char* prefix = "[whisper] ";
                            WriteUTF8(prefix);

                            std::vector<std::string> line;

                            int n = whisper_full_n_segments(ctx);
                            for (int i = 0; i < n; ++i) {
                                const char* text_utf8 = whisper_full_get_segment_text(ctx, i);
                                if (text_utf8 && text_utf8[0]) {
                                    WriteUTF8(text_utf8);
                                    line.push_back(text_utf8);
                                }
                            }
                            WriteUTF8("\n");

                            if (line.size() > 0) {
                                std::lock_guard<std::mutex> lk(smutex);
                               //^^^^^^^^^^^^^^^^^^ speak_queue.push(std::move(line));
                                scv.notify_one();
                            }
                        }

                    }
                }
               

                std::cout << "=================================================================================================================" << std::endl;

                recording_started = false;
            }
        }
        else {
            silence_count = 0;
            active_count++;
            recording_started = true;
        }
    }

    whisper_free(ctx);
}

bool RecordThrede() {
    HRESULT hr = CoInitializeEx(NULL, COINIT_MULTITHREADED);
    if (FAILED(hr)) return false;

    IMMDeviceEnumerator* pEnum = nullptr;
    hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), NULL, CLSCTX_ALL,
        __uuidof(IMMDeviceEnumerator), (void**)&pEnum);
    if (FAILED(hr)) { CoUninitialize(); return false; }

    IMMDevice* pDevice = nullptr;
    hr = pEnum->GetDefaultAudioEndpoint(eCapture, eConsole, &pDevice);
    if (FAILED(hr)) { pEnum->Release(); CoUninitialize(); return false; }

    IAudioClient* pAudioClient = nullptr;
    hr = pDevice->Activate(__uuidof(IAudioClient), CLSCTX_ALL, NULL, (void**)&pAudioClient);
    if (FAILED(hr)) { pDevice->Release(); pEnum->Release(); CoUninitialize(); return false; }

    WAVEFORMATEX* pwfx = nullptr;
    hr = pAudioClient->GetMixFormat(&pwfx);
    if (FAILED(hr)) { pAudioClient->Release(); pDevice->Release(); pEnum->Release(); CoUninitialize(); return false; }

    const int source_channels = pwfx->nChannels;
    const int src_rate = pwfx->nSamplesPerSec;
    hr = pAudioClient->Initialize(AUDCLNT_SHAREMODE_SHARED, 0, 10000000, 0, pwfx, NULL);
    if (FAILED(hr)) { CoTaskMemFree(pwfx); pAudioClient->Release(); pDevice->Release(); pEnum->Release(); CoUninitialize(); return false; }

    IAudioCaptureClient* pCapture = nullptr;
    hr = pAudioClient->GetService(__uuidof(IAudioCaptureClient), (void**)&pCapture);
    if (FAILED(hr)) { CoTaskMemFree(pwfx); pAudioClient->Release(); pDevice->Release(); pEnum->Release(); CoUninitialize(); return false; }

    hr = pAudioClient->Start();
    if (FAILED(hr)) { pCapture->Release(); CoTaskMemFree(pwfx); pAudioClient->Release(); pDevice->Release(); pEnum->Release(); CoUninitialize(); return false; }

    std::cout << "Recording... press Enter to stop.\n";

    const long chunk_frames_src = static_cast<long>(std::ceil(CHUNK_SECONDS * src_rate));
    std::vector<float> chunk_src;
    chunk_src.reserve(chunk_frames_src * source_channels);

    while (!stop_flag) {
        UINT32 packetLength = 0;
        hr = pCapture->GetNextPacketSize(&packetLength);
        if (FAILED(hr)) break;
        if (packetLength == 0) { Sleep(5); continue; }

        BYTE* pData = nullptr;
        UINT32 numFrames = 0;
        DWORD flags = 0;
        hr = pCapture->GetBuffer(&pData, &numFrames, &flags, NULL, NULL);
        if (FAILED(hr)) break;

        if (flags & AUDCLNT_BUFFERFLAGS_SILENT) {
            for (UINT32 f = 0; f < numFrames; ++f)
                for (int ch = 0; ch < source_channels; ++ch)
                    chunk_src.push_back(0.0f);
        }
        else {
            if (pwfx->wBitsPerSample == 32 && (pwfx->wFormatTag == WAVE_FORMAT_IEEE_FLOAT || pwfx->wFormatTag == WAVE_FORMAT_EXTENSIBLE)) {
                const float* samples = reinterpret_cast<const float*>(pData);
                for (UINT32 f = 0; f < numFrames; ++f)
                    for (int ch = 0; ch < source_channels; ++ch)
                        chunk_src.push_back(samples[f * source_channels + ch]);
            }
            else if (pwfx->wBitsPerSample == 16 && pwfx->wFormatTag == WAVE_FORMAT_PCM) {
                const int16_t* samples = reinterpret_cast<const int16_t*>(pData);
                for (UINT32 f = 0; f < numFrames; ++f)
                    for (int ch = 0; ch < source_channels; ++ch)
                        chunk_src.push_back(std::clamp(float(samples[f * source_channels + ch]) / 32768.0f, -1.0f, 1.0f));
            }
        }

        pCapture->ReleaseBuffer(numFrames);

        size_t frames_available = chunk_src.size() / source_channels;
        while (frames_available >= (size_t)chunk_frames_src) {
            std::vector<float> src_chunk_mono;
            src_chunk_mono.reserve(chunk_frames_src);
            for (long f = 0; f < chunk_frames_src; ++f) {
                float sum = 0.0f;
                for (int ch = 0; ch < source_channels; ++ch)
                    sum += chunk_src[f * source_channels + ch];
                src_chunk_mono.push_back(sum / source_channels);
            }
            size_t remove_samples = chunk_frames_src * source_channels;
            chunk_src.erase(chunk_src.begin(), chunk_src.begin() + remove_samples);
            frames_available = chunk_src.size() / source_channels;

            std::vector<float> chunk16;
            if (src_rate != TARGET_RATE)
                chunk16 = resample_float_mono(src_chunk_mono, src_rate, TARGET_RATE);
            else
                chunk16 = std::move(src_chunk_mono);

            if (chunk16.empty()) continue;
            {
                std::lock_guard<std::mutex> lk(qmutex);
                audio_queue.push(std::move(chunk16));
            }
            qcv.notify_one();
        }
    }

    pAudioClient->Stop();
    stop_flag = true;
    qcv.notify_all();
    scv.notify_all();

    CoTaskMemFree(pwfx);
    pCapture->Release();
    pAudioClient->Release();
    pDevice->Release();
    pEnum->Release();
    CoUninitialize();
    return true;
}

static void print_usage(int argc, char** argv) {
    (void)argc;
    LOG("\nexample usage:\n");
    LOG("\n  text generation:     %s -m your_model.gguf -p \"I believe the meaning of life is\" -n 128 -no-cnv\n", argv[0]);
    LOG("\n  chat (conversation): %s -m your_model.gguf -sys \"You are a helpful assistant\"\n", argv[0]);
    LOG("\n");
}

static bool file_exists(const std::string& path) {
    std::ifstream f(path.c_str());
    return f.good();
}

static bool file_is_empty(const std::string& path) {
    std::ifstream f;
    f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    f.open(path.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
    return f.tellg() == 0;
}

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
static void sigint_handler(int signo) {
    if (signo == SIGINT) {
        if (!is_interacting && g_params->interactive) {
            is_interacting = true;
            need_insert_eot = true;
        }
        else {
            console::cleanup();
            LOG("\n");
            common_perf_print(*g_ctx, *g_smpl);
            LOG("Interrupted by user\n");
            common_log_pause(common_log_main());
            _exit(130);
        }
    }
}
#endif

// ================================
   // ★ 5 回録音して平均化して話者登録する関数
   //    enrollSpeakerMulti5()
   // ================================
bool enrollSpeakerMulti5(SpeakerIdentifier& identifier,
    const std::string& speakerName,
    const std::vector<std::vector<float>>& audioList)
{
    if (audioList.size() != 5) {
        std::cerr << "[Error] audioList must contain exactly 5 recordings." << std::endl;
        return false;
    }

    std::vector<std::vector<float>> embeddings;
    embeddings.reserve(5);

    // 5回分の埋め込みを生成
    for (int i = 0; i < 5; i++) {
        const auto& audio = audioList[i];
        if (audio.empty()) {
            std::cerr << "[Error] One of the recordings is empty." << std::endl;
            return false;
        }

        auto mel = identifier.melExtractor_->extract(audio);
        auto emb = identifier.computeEmbeddingByChunks(mel);

        if (emb.empty()) {
            std::cerr << "[Error] Failed to extract embedding (#" << i << ")." << std::endl;
            return false;
        }

        // 正規化
        identifier.l2normalize(emb);
        embeddings.push_back(std::move(emb));
    }

    // ================================
    // ★ 5つの埋め込みを平均化
    // ================================
    std::vector<float> avg(embeddings[0].size(), 0.0f);

    for (size_t d = 0; d < avg.size(); d++) {
        float sum = 0.0f;
        for (int i = 0; i < 5; i++) sum += embeddings[i][d];
        avg[d] = sum / 5.0f;
    }

    // 最後にもう一度 L2 正規化
    identifier.l2normalize(avg);

    // ================================
    // ★ SpeakerModel として保存
    // ================================
    SpeakerModel model;
    model.name = speakerName;
    model.embedding = avg;

    std::filesystem::path dir = identifier.modelDir_;
    if (!std::filesystem::exists(dir)) std::filesystem::create_directories(dir);

    std::filesystem::path file = dir / std::filesystem::u8path(speakerName + ".bin");

    if (!model.saveToFile(file)) {
        std::cerr << "[Error] Failed to save averaged speaker model." << std::endl;
        return false;
    }

    // メモリ上のリストへ反映
    bool found = false;
    for (auto& spk : identifier.speakers_) {
        if (spk.name == speakerName) {
            spk = model;
            found = true;
            break;
        }
    }
    if (!found) identifier.speakers_.push_back(model);

    std::cout << "[SpeakerID] Enrolled (5‑recording averaged): " << speakerName << std::endl;
    return true;
}


int main(int argc, char** argv) {

    InitializeConsole();

    if (g_voice.Init() != 0) {
        WriteUTF8("Failed to initialize voice engine.\n");
        return 1;
    }

    common_params params;
    g_params = &params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MAIN, print_usage)) {
        return 1;
    }

    params.model.path = "D:/work/models/gemma-3-4b-it-Q4_K_M.gguf";

    std::ifstream system_prompt_file("D:\\work\\llama\\LamaCli\\LamaCli\\system_prompt.txt");
    if (system_prompt_file) {
        std::stringstream buffer;
        buffer << system_prompt_file.rdbuf();
        params.system_prompt = buffer.str();
        if (params.system_prompt.rfind("\xEF\xBB\xBF", 0) == 0) {
            params.system_prompt.erase(0, 3);
        }
        LOG_INF("Loaded system prompt from file: %s", params.system_prompt.c_str());
    }
    else {
        LOG_WRN("Could not open system_prompt.txt. Using hardcoded system prompt.");
        params.system_prompt = "あなたは誠実で優秀な日本人のアシスタントです。";
    }

    common_init();

    auto& sparams = params.sampling;

    console::init(params.simple_io, params.use_color);
    atexit([]() { console::cleanup(); });

    if (params.embedding) {
        LOG_ERR("************\n");
        LOG_ERR("%s: please use the 'embedding' tool for embedding calculations\n", __func__);
        LOG_ERR("************\n\n");
        return 0;
    }

    if (params.n_ctx != 0 && params.n_ctx < 8) {
        LOG_WRN("%s: warning: minimum context size is 8, using minimum size.\n", __func__);
        params.n_ctx = 8;
    }

    if (params.rope_freq_base != 0.0) {
        LOG_WRN("%s: warning: changing RoPE frequency base to %g.\n", __func__, params.rope_freq_base);
    }

    if (params.rope_freq_scale != 0.0) {
        LOG_WRN("%s: warning: scaling RoPE frequency by %g.\n", __func__, params.rope_freq_scale);
    }

    LOG_INF("%s: llama backend init\n", __func__);

    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    common_sampler* smpl = nullptr;

    g_model = &model;
    g_ctx = &ctx;
    g_smpl = &smpl;

    std::vector<common_chat_msg> chat_msgs;
    {
        common_chat_msg msg;
        msg.role = "user";
        msg.content = "こんにちは。";
        chat_msgs.push_back(msg);

        msg.role = "assistant";
        msg.content = "こんにちは。何かお手伝いできることはありますか？";
        chat_msgs.push_back(msg);
    }

    LOG_INF("%s: load the model and apply lora adapter, if any\n", __func__);
    common_init_result llama_init = common_init_from_params(params);

    model = llama_init.model.get();
    ctx = llama_init.context.get();

    if (model == NULL) {
        LOG_ERR("%s: error: unable to load model\n", __func__);
        return 1;
    }

    auto* mem = llama_get_memory(ctx);

    const llama_vocab* vocab = llama_model_get_vocab(model);
    auto chat_templates = common_chat_templates_init(model, params.chat_template);

    LOG_INF("%s: llama threadpool init, n_threads = %d\n", __func__, (int)params.cpuparams.n_threads);

    auto* cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu_dev) {
        LOG_ERR("%s: no CPU backend found\n", __func__);
        return 1;
    }
    auto* reg = ggml_backend_dev_backend_reg(cpu_dev);
    auto* ggml_threadpool_new_fn = (decltype(ggml_threadpool_new)*)ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new");
    auto* ggml_threadpool_free_fn = (decltype(ggml_threadpool_free)*)ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_free");

    struct ggml_threadpool_params tpp_batch =
        ggml_threadpool_params_from_cpu_params(params.cpuparams_batch);
    struct ggml_threadpool_params tpp =
        ggml_threadpool_params_from_cpu_params(params.cpuparams);

    set_process_priority(params.cpuparams.priority);

    struct ggml_threadpool* threadpool_batch = NULL;
    if (!ggml_threadpool_params_match(&tpp, &tpp_batch)) {
        threadpool_batch = ggml_threadpool_new_fn(&tpp_batch);
        if (!threadpool_batch) {
            LOG_ERR("%s: batch threadpool create failed : n_threads %d\n", __func__, tpp_batch.n_threads);
            return 1;
        }
        tpp.paused = true;
    }

    struct ggml_threadpool* threadpool = ggml_threadpool_new_fn(&tpp);
    if (!threadpool) {
        LOG_ERR("%s: threadpool create failed : n_threads %d\n", __func__, tpp.n_threads);
        return 1;
    }

    llama_attach_threadpool(ctx, threadpool, threadpool_batch);

    const int n_ctx_train = llama_model_n_ctx_train(model);
    const int n_ctx = llama_n_ctx(ctx);

    if (n_ctx > n_ctx_train) {
        LOG_WRN("%s: model was trained on only %d context tokens (%d specified)\n", __func__, n_ctx_train, n_ctx);
    }

    const bool has_chat_template = common_chat_templates_was_explicit(chat_templates.get());
    if (params.conversation_mode == COMMON_CONVERSATION_MODE_AUTO) {
        if (has_chat_template) {
            LOG_INF("%s: chat template is available, enabling conversation mode (disable it with -no-cnv)\n", __func__);
            params.conversation_mode = COMMON_CONVERSATION_MODE_ENABLED;
        }
        else {
            params.conversation_mode = COMMON_CONVERSATION_MODE_DISABLED;
        }
    }

    if (params.conversation_mode && !has_chat_template) {
        LOG_WRN("%s: chat template is not available or is not supported. This may cause the model to output suboptimal responses\n", __func__);
    }

    if (params.conversation_mode) {
        if (params.enable_chat_template) {
            if (!params.prompt.empty() && params.system_prompt.empty()) {
                LOG_WRN("*** User-specified prompt will pre-start conversation, did you mean to set --system-prompt (-sys) instead?\n");
            }

            LOG_INF("%s: chat template example:\n%s\n", __func__, common_chat_format_example(chat_templates.get(), params.use_jinja, params.default_template_kwargs).c_str());
        }
        else {
            LOG_INF("%s: in-suffix/prefix is specified, chat template will be disabled\n", __func__);
        }
    }

    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
        LOG_INF("\n");
    }

    std::string path_session = params.path_prompt_cache;
    std::vector<llama_token> session_tokens;

    if (!path_session.empty()) {
        LOG_INF("%s: attempting to load saved session from '%s'\n", __func__, path_session.c_str());
        if (!file_exists(path_session)) {
            LOG_INF("%s: session file does not exist, will create.\n", __func__);
        }
        else if (file_is_empty(path_session)) {
            LOG_INF("%s: The session file is empty. A new session will be initialized.\n", __func__);
        }
        else {
            session_tokens.resize(n_ctx);
            size_t n_token_count_out = 0;
            if (!llama_state_load_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.capacity(), &n_token_count_out)) {
                LOG_ERR("%s: failed to load session file '%s'\n", __func__, path_session.c_str());
                return 1;
            }
            session_tokens.resize(n_token_count_out);
            LOG_INF("%s: loaded a session with prompt size of %d tokens\n", __func__, (int)session_tokens.size());
        }
    }

    const bool add_bos = llama_vocab_get_add_bos(vocab) && !params.use_jinja;
    if (!llama_model_has_encoder(model)) {
        GGML_ASSERT(!llama_vocab_get_add_eos(vocab));
    }

    LOG_DBG("n_ctx: %d, add_bos: %d\n", n_ctx, add_bos);

    std::vector<llama_token> embd_inp;

    bool waiting_for_first_input = false;
    auto chat_add_and_format = [&chat_msgs, &chat_templates](const std::string& role, const std::string& content) {
        common_chat_msg new_msg;
        new_msg.role = role;
        new_msg.content = content;
        auto formatted = common_chat_format_single(chat_templates.get(), chat_msgs, new_msg, role == "user", g_params->use_jinja);
        chat_msgs.push_back(new_msg);
        LOG_DBG("formatted: '%s'\n", formatted.c_str());
        return formatted;
        };

    std::string prompt;
    {
        if (params.conversation_mode && params.enable_chat_template) {
            if (!params.system_prompt.empty()) {
                chat_add_and_format("system", params.system_prompt);
            }

            if (!params.prompt.empty()) {
                chat_add_and_format("user", params.prompt);
            }
            else {
                waiting_for_first_input = true;
            }

            if (!params.system_prompt.empty() || !params.prompt.empty()) {
                common_chat_templates_inputs inputs;
                inputs.use_jinja = g_params->use_jinja;
                inputs.messages = chat_msgs;
                inputs.add_generation_prompt = !params.prompt.empty();

                prompt = common_chat_templates_apply(chat_templates.get(), inputs).prompt;

                LOG_INF("Final prompt to model:\n---\n%s\n---\n", prompt.c_str());
            }
        }
        else {
            prompt = params.prompt;
        }

        if (params.interactive_first || !prompt.empty() || session_tokens.empty()) {
            LOG_DBG("tokenize the prompt\n");
            embd_inp = common_tokenize(ctx, prompt, true, true);
        }
        else {
            LOG_DBG("use session tokens\n");
            embd_inp = session_tokens;
        }

        LOG_DBG("prompt: \"%s\"\n", prompt.c_str());
        LOG_DBG("tokens: %s\n", string_from(ctx, embd_inp).c_str());
    }

    if (!waiting_for_first_input && embd_inp.empty()) {
        if (add_bos) {
            embd_inp.push_back(llama_vocab_bos(vocab));
            LOG_WRN("embd_inp was considered empty and bos was added: %s\n", string_from(ctx, embd_inp).c_str());
        }
        else {
            LOG_ERR("input is empty\n");
            return -1;
        }
    }

    if ((int)embd_inp.size() > n_ctx - 4) {
        LOG_ERR("%s: prompt is too long (%d tokens, max %d)\n", __func__, (int)embd_inp.size(), n_ctx - 4);
        return 1;
    }

    size_t n_matching_session_tokens = 0;
    if (!session_tokens.empty()) {
        for (llama_token id : session_tokens) {
            if (n_matching_session_tokens >= embd_inp.size() || id != embd_inp[n_matching_session_tokens]) {
                break;
            }
            n_matching_session_tokens++;
        }
        if (params.prompt.empty() && n_matching_session_tokens == embd_inp.size()) {
            LOG_INF("%s: using full prompt from session file\n", __func__);
        }
        else if (n_matching_session_tokens >= embd_inp.size()) {
            LOG_INF("%s: session file has exact match for prompt!\n", __func__);
        }
        else if (n_matching_session_tokens < (embd_inp.size() / 2)) {
            LOG_WRN("%s: session file has low similarity to prompt (%zu / %zu tokens); will mostly be reevaluated\n",
                __func__, n_matching_session_tokens, embd_inp.size());
        }
        else {
            LOG_INF("%s: session file matches %zu / %zu tokens of prompt\n",
                __func__, n_matching_session_tokens, embd_inp.size());
        }

        llama_memory_seq_rm(mem, -1, n_matching_session_tokens, -1);
    }

    LOG_DBG("recalculate the cached logits (check): embd_inp.size() %zu, n_matching_session_tokens %zu, embd_inp.size() %zu, session_tokens.size() %zu\n",
        embd_inp.size(), n_matching_session_tokens, embd_inp.size(), session_tokens.size());

    if (!embd_inp.empty() && n_matching_session_tokens == embd_inp.size() && session_tokens.size() > embd_inp.size()) {
        LOG_DBG("recalculate the cached logits (do): session_tokens.resize( %zu )\n", embd_inp.size() - 1);
        session_tokens.resize(embd_inp.size() - 1);
    }

    if (params.n_keep < 0 || params.n_keep >(int)embd_inp.size()) {
        params.n_keep = (int)embd_inp.size();
    }
    else {
        params.n_keep += add_bos;
    }

    if (params.conversation_mode) {
        if (params.single_turn && !params.prompt.empty()) {
            params.interactive = false;
            params.interactive_first = false;
        }
        else {
            params.interactive_first = true;
        }
    }

    if (params.interactive_first) {
        params.interactive = true;
    }

    if (params.verbose_prompt) {
        LOG_INF("%s: prompt: '%s'\n", __func__, params.prompt.c_str());
        LOG_INF("%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
        for (int i = 0; i < (int)embd_inp.size(); i++) {
            LOG_INF("%6d -> '%s'\n", embd_inp[i], common_token_to_piece(ctx, embd_inp[i]).c_str());
        }

        if (params.n_keep > add_bos) {
            LOG_INF("%s: static prompt based on n_keep: '", __func__);
            for (int i = 0; i < params.n_keep; i++) {
                LOG_CNT("%s", common_token_to_piece(ctx, embd_inp[i]).c_str());
            }
            LOG_CNT("'\n");
        }
        LOG_INF("\n");
    }

    {
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
        struct sigaction sigint_action;
        sigint_action.sa_handler = sigint_handler;
        sigemptyset(&sigint_action.sa_mask);
        sigint_action.sa_flags = 0;
        sigaction(SIGINT, &sigint_action, NULL);
#elif defined (_WIN32)
        auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
            return (ctrl_type == CTRL_C_EVENT) ? (sigint_handler(SIGINT), true) : false;
            };
        SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif
    }

    if (params.interactive) {
        LOG_INF("%s: interactive mode on.\n", __func__);

        if (!params.antiprompt.empty()) {
            for (const auto& antiprompt : params.antiprompt) {
                LOG_INF("Reverse prompt: '%s'\n", antiprompt.c_str());
                if (params.verbose_prompt) {
                    auto tmp = common_tokenize(ctx, antiprompt, false, true);
                    for (int i = 0; i < (int)tmp.size(); i++) {
                        LOG_INF("%6d -> '%s'\n", tmp[i], common_token_to_piece(ctx, tmp[i]).c_str());
                    }
                }
            }
        }

        if (params.input_prefix_bos) {
            LOG_INF("Input prefix with BOS\n");
        }

        if (!params.input_prefix.empty()) {
            LOG_INF("Input prefix: '%s'\n", params.input_prefix.c_str());
            if (params.verbose_prompt) {
                auto tmp = common_tokenize(ctx, params.input_prefix, true, true);
                for (int i = 0; i < (int)tmp.size(); i++) {
                    LOG_INF("%6d -> '%s'\n", tmp[i], common_token_to_piece(ctx, tmp[i]).c_str());
                }
            }
        }

        if (!params.input_suffix.empty()) {
            LOG_INF("Input suffix: '%s'\n", params.input_suffix.c_str());
            if (params.verbose_prompt) {
                auto tmp = common_tokenize(ctx, params.input_suffix, false, true);
                for (int i = 0; i < (int)tmp.size(); i++) {
                    LOG_INF("%6d -> '%s'\n", tmp[i], common_token_to_piece(ctx, tmp[i]).c_str());
                }
            }
        }
    }

    smpl = common_sampler_init(model, sparams);
    if (!smpl) {
        LOG_ERR("%s: failed to initialize sampling subsystem\n", __func__);
        return 1;
    }

    LOG_INF("sampler seed: %u\n", common_sampler_get_seed(smpl));
    LOG_INF("sampler params: \n%s\n", sparams.print().c_str());
    LOG_INF("sampler chain: %s\n", common_sampler_print(smpl).c_str());

    LOG_INF("generate: n_ctx = %d, n_batch = %d, n_predict = %d, n_keep = %d\n", n_ctx, params.n_batch, params.n_predict, params.n_keep);

    int ga_i = 0;

    const int ga_n = params.grp_attn_n;
    const int ga_w = params.grp_attn_w;

    if (ga_n != 1) {
        GGML_ASSERT(ga_n > 0 && "grp_attn_n must be positive");
        GGML_ASSERT(ga_w % ga_n == 0 && "grp_attn_w must be a multiple of grp_attn_n");
        LOG_INF("self-extend: n_ctx_train = %d, grp_attn_n = %d, grp_attn_w = %d\n", n_ctx_train, ga_n, ga_w);
    }
    LOG_INF("\n");

    if (params.interactive) {
        const char* control_message;
        if (params.multiline_input) {
            control_message = " - To return control to the AI, end your input with '\\'.\n"
                " - To return control without starting a new line, end your input with '/'.\n";
        }
        else {
            control_message = " - Press Return to return control to the AI.\n"
                " - To return control without starting a new line, end your input with '/'.\n"
                " - If you want to submit another line, end your input with '\\'.\n";
        }
        LOG_INF("== Running in interactive mode. ==\n");
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
        LOG_INF(" - Press Ctrl+C to interject at any time.\n");
#endif
        LOG_INF("%s", control_message);
        if (params.conversation_mode && params.enable_chat_template && params.system_prompt.empty()) {
            LOG_INF(" - Not using system message. To change it, set a different value via -sys PROMPT\n");
        }
        LOG_INF("\n");

        is_interacting = params.interactive_first;
    }

    bool is_antiprompt = false;
    bool input_echo = true;
    bool display = true;
    bool need_to_save_session = !path_session.empty() && n_matching_session_tokens < embd_inp.size();

    int n_past = 0;
    int n_remain = params.n_predict;
    int n_consumed = 0;
    int n_session_consumed = 0;

    std::vector<int> input_tokens;
    g_input_tokens = &input_tokens;
    std::vector<int> output_tokens;
    g_output_tokens = &output_tokens;
    std::ostringstream output_ss;
    g_output_ss = &output_ss;
    std::ostringstream assistant_ss;

    console::set_display(console::prompt);
    display = params.display_prompt;

    std::vector<llama_token> embd;

    std::vector<llama_token> antiprompt_token;

    for (const std::string& antiprompt : params.antiprompt) {
        auto ids = ::common_tokenize(ctx, antiprompt, false, true);
        if (ids.size() == 1) {
            antiprompt_token.push_back(ids[0]);
        }
    }

    if (llama_model_has_encoder(model)) {
        int enc_input_size = embd_inp.size();
        llama_token* enc_input_buf = embd_inp.data();

        if (llama_encode(ctx, llama_batch_get_one(enc_input_buf, enc_input_size))) {
            LOG_ERR("%s : failed to eval\n", __func__);
            return 1;
        }

        llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
        if (decoder_start_token_id == LLAMA_TOKEN_NULL) {
            decoder_start_token_id = llama_vocab_bos(vocab);
        }

        embd_inp.clear();
        embd_inp.push_back(decoder_start_token_id);
    }


    std::thread worker1(RecordThrede);
    worker1.detach();

    // 少し待機してスレッドが開始されるのを待つ
    Sleep(100);

    WriteUTF8("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    WriteUTF8("モードを選択:\n");
    WriteUTF8("1. 登録モード (新しい話者を登録)\n");
    WriteUTF8("2. 判定モード (話者を識別)\n");
    WriteUTF8("3. 管理モード (話者削除)\n");
    WriteUTF8("4. 終了\n");
    WriteUTF8("━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    WriteUTF8("\n--- 新しい話者を登録しますか？ ---\n");
    WriteUTF8("Yes (Y) or No (N): ");

    std::string mode1 = ReadUTF8Input();


    if (mode1 == "Y" || mode1 == "y") {
        Enrollment_thread();
    }

    std::thread worker2(worker_thread_func);
    worker2.detach();

    while ((n_remain != 0 && !is_antiprompt) || params.interactive) {
        if (!embd.empty()) {
            int max_embd_size = n_ctx - 4;

            if ((int)embd.size() > max_embd_size) {
                const int skipped_tokens = (int)embd.size() - max_embd_size;
                embd.resize(max_embd_size);

                console::set_display(console::error);
                LOG_WRN("<<input too long: skipped %d token%s>>", skipped_tokens, skipped_tokens != 1 ? "s" : "");
                console::set_display(console::reset);
            }

            if (ga_n == 1) {
                if (n_past + (int)embd.size() >= n_ctx) {
                    if (!params.ctx_shift) {
                        LOG_WRN("\n\n%s: context full and context shift is disabled => stopping\n", __func__);
                        break;
                    }

                    if (params.n_predict == -2) {
                        LOG_WRN("\n\n%s: context full and n_predict == %d => stopping\n", __func__, params.n_predict);
                        break;
                    }

                    const int n_left = n_past - params.n_keep;
                    const int n_discard = n_left / 2;

                    LOG_DBG("context full, swapping: n_past = %d, n_left = %d, n_ctx = %d, n_keep = %d, n_discard = %d\n",
                        n_past, n_left, n_ctx, params.n_keep, n_discard);

                    llama_memory_seq_rm(mem, 0, params.n_keep, params.n_keep + n_discard);
                    llama_memory_seq_add(mem, 0, params.n_keep + n_discard, n_past, -n_discard);

                    n_past -= n_discard;

                    LOG_DBG("after swap: n_past = %d\n", n_past);
                    LOG_DBG("embd: %s\n", string_from(ctx, embd).c_str());
                    LOG_DBG("clear session path\n");
                    path_session.clear();
                }
            }
            else {
                while (n_past >= ga_i + ga_w) {
                    const int ib = (ga_n * ga_i) / ga_w;
                    const int bd = (ga_w / ga_n) * (ga_n - 1);
                    const int dd = (ga_w / ga_n) - ib * bd - ga_w;

                    LOG_DBG("\n");
                    LOG_DBG("shift: [%6d, %6d] + %6d -> [%6d, %6d]\n", ga_i, n_past, ib * bd, ga_i + ib * bd, n_past + ib * bd);
                    LOG_DBG("div:   [%6d, %6d] / %6d -> [%6d, %6d]\n", ga_i + ib * bd, ga_i + ib * bd + ga_w, ga_n, (ga_i + ib * bd) / ga_n, (ga_i + ib * bd + ga_w) / ga_n);
                    LOG_DBG("shift: [%6d, %6d] + %6d -> [%6d, %6d]\n", ga_i + ib * bd + ga_w, n_past + ib * bd, dd, ga_i + ib * bd + ga_w + dd, n_past + ib * bd + dd);

                    llama_memory_seq_add(mem, 0, ga_i, n_past, ib * bd);
                    llama_memory_seq_div(mem, 0, ga_i + ib * bd, ga_i + ib * bd + ga_w, ga_n);
                    llama_memory_seq_add(mem, 0, ga_i + ib * bd + ga_w, n_past + ib * bd, dd);

                    n_past -= bd;
                    ga_i += ga_w / ga_n;

                    LOG_DBG("\nn_past_old = %d, n_past = %d, ga_i = %d\n\n", n_past + bd, n_past, ga_i);
                }
            }

            if (n_session_consumed < (int)session_tokens.size()) {
                size_t i = 0;
                for (; i < embd.size(); i++) {
                    if (embd[i] != session_tokens[n_session_consumed]) {
                        session_tokens.resize(n_session_consumed);
                        break;
                    }

                    n_past++;
                    n_session_consumed++;

                    if (n_session_consumed >= (int)session_tokens.size()) {
                        ++i;
                        break;
                    }
                }
                if (i > 0) {
                    embd.erase(embd.begin(), embd.begin() + i);
                }
            }

            for (int i = 0; i < (int)embd.size(); i += params.n_batch) {
                int n_eval = (int)embd.size() - i;
                if (n_eval > params.n_batch) {
                    n_eval = params.n_batch;
                }

                LOG_DBG("eval: %s\n", string_from(ctx, embd).c_str());

                if (llama_decode(ctx, llama_batch_get_one(&embd[i], n_eval))) {
                    LOG_ERR("%s : failed to eval\n", __func__);
                    return 1;
                }

                n_past += n_eval;

                LOG_DBG("n_past = %d\n", n_past);
                if (params.n_print > 0 && n_past % params.n_print == 0) {
                    LOG_DBG("\n\033[31mTokens consumed so far = %d / %d \033[0m\n", n_past, n_ctx);
                }
            }

            if (!embd.empty() && !path_session.empty()) {
                session_tokens.insert(session_tokens.end(), embd.begin(), embd.end());
                n_session_consumed = session_tokens.size();
            }
        }

        embd.clear();

        if ((int)embd_inp.size() <= n_consumed && !is_interacting) {
            const llama_token id = common_sampler_sample(smpl, ctx, -1);

            common_sampler_accept(smpl, id, true);

            embd.push_back(id);

            input_echo = true;

            --n_remain;

            LOG_DBG("n_remain: %d\n", n_remain);
        }
        else {
            LOG_DBG("embd_inp.size(): %d, n_consumed: %d\n", (int)embd_inp.size(), n_consumed);
            while ((int)embd_inp.size() > n_consumed) {
                embd.push_back(embd_inp[n_consumed]);

                common_sampler_accept(smpl, embd_inp[n_consumed], false);

                ++n_consumed;
                if ((int)embd.size() >= params.n_batch) {
                    break;
                }
            }
        }

        if (input_echo && display) {
            for (auto id : embd) {
                const std::string token_str = common_token_to_piece(ctx, id, params.special);

                LOG("%s", token_str.c_str());

                if (embd.size() > 1) {
                    input_tokens.push_back(id);
                }
                else {
                    output_tokens.push_back(id);
                    output_ss << token_str;
                }
            }
        }

        if (input_echo && (int)embd_inp.size() == n_consumed) {
            console::set_display(console::reset);
            display = true;
        }

        if ((int)embd_inp.size() <= n_consumed) {
            if (!params.antiprompt.empty()) {
                const int n_prev = 32;
                const std::string last_output = common_sampler_prev_str(smpl, ctx, n_prev);

                is_antiprompt = false;
                for (std::string& antiprompt : params.antiprompt) {
                    size_t extra_padding = params.interactive ? 0 : 2;
                    size_t search_start_pos = last_output.length() > static_cast<size_t>(antiprompt.length() + extra_padding)
                        ? last_output.length() - static_cast<size_t>(antiprompt.length() + extra_padding)
                        : 0;

                    if (last_output.find(antiprompt, search_start_pos) != std::string::npos) {
                        if (params.interactive) {
                            is_interacting = true;
                        }
                        is_antiprompt = true;
                        break;
                    }
                }

                if (!last_output.empty()) {
                    llama_token last_token = common_sampler_last(smpl);
                    for (auto token : antiprompt_token) {
                        if (token == last_token) {
                            if (params.interactive) {
                                is_interacting = true;
                            }
                            is_antiprompt = true;
                            break;
                        }
                    }
                }

                if (is_antiprompt) {
                    LOG_DBG("found antiprompt: %s\n", last_output.c_str());
                }
            }

            if (!waiting_for_first_input && llama_vocab_is_eog(vocab, common_sampler_last(smpl))) {
                LOG_DBG("found an EOG token\n");

                if (params.interactive) {
                    if (!params.antiprompt.empty()) {
                        const auto first_antiprompt = common_tokenize(ctx, params.antiprompt.front(), false, true);
                        embd_inp.insert(embd_inp.end(), first_antiprompt.begin(), first_antiprompt.end());
                        is_antiprompt = true;
                    }

                    if (params.enable_chat_template) {
                        chat_add_and_format("assistant", assistant_ss.str());
                    }

                    if (!assistant_ss.str().empty()) {
                        g_voice.VoicePlay(assistant_ss.str());
                    }

                    is_interacting = true;
                    LOG("\n");
                }
            }

            if (params.conversation_mode && !waiting_for_first_input) {
                const auto id = common_sampler_last(smpl);
                if (!llama_vocab_is_eog(vocab, id)) {
                    assistant_ss << common_token_to_piece(ctx, id, false);
                }

                if (!prompt.empty()) {
                    prompt.clear();
                    is_interacting = false;
                }
            }

            if ((n_past > 0 || waiting_for_first_input) && is_interacting) {
                LOG_DBG("waiting for user input\n");

                if (params.conversation_mode) {
                    LOG("\n> ");
                }

                if (params.input_prefix_bos) {
                    LOG_DBG("adding input prefix BOS token\n");
                    embd_inp.push_back(llama_vocab_bos(vocab));
                }

                if (!params.input_prefix.empty() && !params.conversation_mode) {
                    LOG_DBG("appending input prefix: '%s'\n", params.input_prefix.c_str());
                    LOG("%s", params.input_prefix.c_str());
                }

                console::set_display(console::user_input);
                display = params.display_prompt;

                std::string buffer;
                {
                    std::unique_lock<std::mutex> lk(smutex);
                    scv.wait(lk, [] { return !speak_queue.empty(); });
                    auto chunk = std::move(speak_queue.front());

                    speak_queue.pop();
                    lk.unlock();

                    for (const auto& str : chunk) {
                        buffer += str;
                    }
                }

                console::set_display(console::reset);
                display = true;

                if (buffer.empty()) {
                    LOG("EOF by user\n");
                    break;
                }

                if (buffer.back() == '\n') {
                    buffer.pop_back();
                }

                if (buffer.empty()) {
                    LOG_DBG("empty line, passing control back\n");
                }
                else {
                    if (!params.input_suffix.empty() && !params.conversation_mode) {
                        LOG_DBG("appending input suffix: '%s'\n", params.input_suffix.c_str());
                        LOG("%s", params.input_suffix.c_str());
                    }

                    LOG_DBG("buffer: '%s'\n", buffer.c_str());

                    const size_t original_size = embd_inp.size();

                    if (params.escape) {
                        string_process_escapes(buffer);
                    }

                    bool format_chat = params.conversation_mode && params.enable_chat_template;
                    std::string user_inp = format_chat
                        ? chat_add_and_format("user", std::move(buffer))
                        : std::move(buffer);

                    const auto line_pfx = common_tokenize(ctx, params.input_prefix, false, true);
                    const auto line_inp = common_tokenize(ctx, user_inp, false, format_chat);
                    const auto line_sfx = common_tokenize(ctx, params.input_suffix, false, true);

                    LOG_DBG("input tokens: %s\n", string_from(ctx, line_inp).c_str());

                    if (need_insert_eot && format_chat) {
                        llama_token eot = llama_vocab_eot(vocab);
                        embd_inp.push_back(eot == LLAMA_TOKEN_NULL ? llama_vocab_eos(vocab) : eot);
                        need_insert_eot = false;
                    }

                    embd_inp.insert(embd_inp.end(), line_pfx.begin(), line_pfx.end());
                    embd_inp.insert(embd_inp.end(), line_inp.begin(), line_inp.end());
                    embd_inp.insert(embd_inp.end(), line_sfx.begin(), line_sfx.end());

                    if (params.verbose_prompt) {
                        LOG_INF("%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size() - original_size);
                    }
                    std::string str;

                    for (size_t i = original_size; i < embd_inp.size(); ++i) {
                        const llama_token token = embd_inp[i];
                        const std::string token_str = common_token_to_piece(ctx, token);
                        output_tokens.push_back(token);
                        output_ss << token_str;
                        str += token_str;

                        if (params.verbose_prompt) {
                            LOG_INF("%6d -> '%s'\n", token, token_str.c_str());
                        }
                    }

                    assistant_ss.str("");

                    n_remain -= line_inp.size();
                    LOG_DBG("n_remain: %d\n", n_remain);
                }

                input_echo = false;
            }

            if (n_past > 0 || waiting_for_first_input) {
                if (is_interacting) {
                    common_sampler_reset(smpl);
                }
                is_interacting = false;

                if (waiting_for_first_input && params.single_turn) {
                    params.interactive = false;
                    params.interactive_first = false;
                }
                waiting_for_first_input = false;
            }
        }

        if (!embd.empty() && llama_vocab_is_eog(vocab, embd.back()) && !(params.interactive)) {
            LOG(" [end of text]\n");
            break;
        }

        if (params.interactive && n_remain <= 0 && params.n_predict >= 0) {
            n_remain = params.n_predict;
            is_interacting = true;
        }
    }

    LOG("\n\n");
    common_perf_print(ctx, smpl);

    common_sampler_free(smpl);

    llama_backend_free();

    ggml_threadpool_free_fn(threadpool);
    ggml_threadpool_free_fn(threadpool_batch);

    return 0;
}
