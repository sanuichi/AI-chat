// SpeakerIdentifier.h - 性能改善版
// 入力形式: float32[B,T,80] 対応

#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <memory>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <chrono>
#include <iomanip>

#include <Eigen/Dense>
#include <fftw3.h>
#include <onnxruntime_cxx_api.h>
//#include <fftw3_threads.h> 



#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace SpeakerID {

    namespace fs = std::filesystem;

    struct Config {
        int sampleRate = 16000;
        int nMels = 80;
        int fftSize = 1024;
        int hopSize = 160;
        int cudaDeviceId = 0;
        bool useFP16 = false;
        bool useCPUFallback = true;
        double minConfidence = 0.70;
        int chunkFrames = 300;
        int minAudioSeconds = 2;  // 最小音声長（秒）
    };

    struct IdentificationResult {
        std::string speakerName;
        double confidence;
        bool isUnknown;
        std::vector<std::pair<std::string, double>> allScores;

        std::string getConfidenceLevel() const {
            if (confidence >= 0.85) return "very_high";
            if (confidence >= 0.75) return "high";
            if (confidence >= 0.65) return "medium";
            if (confidence >= 0.55) return "low";
            return "very_low";
        }

        std::string getConfidenceDescription() const {
            return getConfidenceLevel();
        }
    };

    // ==================== MelSpectrogramExtractor (修正版) ====================
    class MelSpectrogramExtractor {
    public:
        MelSpectrogramExtractor(const Config& config)
            : sampleRate_(config.sampleRate), nMels_(config.nMels),
            fftSize_(config.fftSize), hopSize_(config.hopSize)
        {
            fftIn_ = (double*)fftw_malloc(sizeof(double) * fftSize_);
            fftOut_ = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (fftSize_ / 2 + 1));
            fftPlan_ = fftw_plan_dft_r2c_1d(fftSize_, fftIn_, fftOut_, FFTW_ESTIMATE);
            melFilterbanks_ = createMelFilterbanks();
            window_ = createHammingWindow();
        }

        ~MelSpectrogramExtractor() {
            if (fftPlan_) fftw_destroy_plan(fftPlan_);
            if (fftIn_) fftw_free(fftIn_);
            if (fftOut_) fftw_free(fftOut_);
        }

        //fftw_init_threads();
        //fftw_plan_with_nthreads(std::thread::hardware_concurrency());


        Eigen::MatrixXd extract(const std::vector<float>& audio_in) {
            if (audio_in.empty()) return Eigen::MatrixXd::Zero(nMels_, 1);

            std::vector<float> audio = audio_in;

            // プリエンファシス
            const float preemphasis = 0.97f;
            for (int i = (int)audio.size() - 1; i > 0; --i) {
                audio[i] -= preemphasis * audio[i - 1];
            }

            // RMS正規化（音量を統一）
            double rms = 0.0;
            for (auto s : audio) rms += s * s;
            rms = std::sqrt(rms / audio.size());

            if (rms > 1e-8) {
                float scale = 0.1f / rms;
                for (auto& s : audio) {
                    s *= scale;
                    s = std::max(-1.0f, std::min(1.0f, s));
                }
            }

            if ((int)audio.size() < fftSize_) {
                audio.resize(fftSize_, 0.0f);
            }

            int numFrames = static_cast<int>((audio.size() - fftSize_) / hopSize_) + 1;
            if (numFrames <= 0) numFrames = 1;

            Eigen::MatrixXd melSpec(nMels_, numFrames);

            // STFT + メルフィルタバンク
            for (int i = 0; i < numFrames; ++i) {
                int start = i * hopSize_;

                for (int j = 0; j < fftSize_; ++j) {
                    double v = 0.0;
                    if (start + j < (int)audio.size()) v = audio[start + j];
                    fftIn_[j] = v * window_[j];
                }

                fftw_execute(fftPlan_);

                Eigen::VectorXd power(fftSize_ / 2 + 1);
                for (int j = 0; j < (fftSize_ / 2 + 1); ++j) {
                    double re = fftOut_[j][0];
                    double im = fftOut_[j][1];
                    power(j) = re * re + im * im;
                }

                Eigen::VectorXd mel = melFilterbanks_ * power;
                //mel = (mel.array() + 1e-10).log();
				mel = (mel.array() + 1e-10).log10() * 10.0;
                melSpec.col(i) = mel;
            }

            // ✅ 全体で正規化（話者特徴を保持）
            double global_mean = melSpec.mean();
            double global_std = 0.0;

            for (int i = 0; i < melSpec.rows(); ++i) {
                for (int j = 0; j < melSpec.cols(); ++j) {
                    double diff = melSpec(i, j) - global_mean;
                    global_std += diff * diff;
                }
            }
            //global_std = std::sqrt(global_std / (melSpec.rows() * melSpec.cols()));
            global_std = std::sqrt((melSpec.array() - global_mean).square().mean());

            if (global_std > 1e-8) {
                melSpec = (melSpec.array() - global_mean) / global_std;
            }

            return melSpec;
        }

    private:
        int sampleRate_, nMels_, fftSize_, hopSize_;
        double* fftIn_ = nullptr;
        fftw_complex* fftOut_ = nullptr;
        fftw_plan fftPlan_ = nullptr;
        std::vector<double> window_;
        Eigen::MatrixXd melFilterbanks_;

        std::vector<double> createHammingWindow() {
            std::vector<double> w(fftSize_);
            for (int i = 0; i < fftSize_; ++i)
                w[i] = 0.54 - 0.46 * std::cos(2.0 * M_PI * i / (fftSize_ - 1));
            return w;
        }

        Eigen::MatrixXd createMelFilterbanks() {
            int nfft = fftSize_ / 2 + 1;
            Eigen::MatrixXd fb = Eigen::MatrixXd::Zero(nMels_, nfft);

            auto hz2mel = [](double hz) { return 2595.0 * std::log10(1.0 + hz / 700.0); };
            auto mel2hz = [](double mel) { return 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0); };

            double melMin = hz2mel(0);
            double melMax = hz2mel(sampleRate_ / 2.0);

            std::vector<double> melPts(nMels_ + 2);
            for (int i = 0; i < nMels_ + 2; ++i)
                melPts[i] = melMin + (melMax - melMin) * i / (nMels_ + 1);

            std::vector<int> bins(nMels_ + 2);
            for (int i = 0; i < nMels_ + 2; ++i) {
                double hz = mel2hz(melPts[i]);
                bins[i] = std::min(nfft - 1, std::max(0,
                    static_cast<int>(std::floor((fftSize_ + 1) * hz / sampleRate_))));
            }

            for (int i = 0; i < nMels_; ++i) {
                int left = bins[i], center = bins[i + 1], right = bins[i + 2];
                if (center == left) center = left + 1;
                if (right == center) right = center + 1;
                center = std::min(center, nfft - 1);
                right = std::min(right, nfft - 1);

                for (int j = left; j < center; ++j)
                    if (j >= 0 && j < nfft)
                        fb(i, j) = (j - left) / double(center - left);

                for (int j = center; j < right; ++j)
                    if (j >= 0 && j < nfft)
                        fb(i, j) = (right - j) / double(right - center);
            }

            return fb;
        }
    };

    // ==================== StatisticalEmbedding ====================
    class StatisticalEmbedding {
    public:
        static std::vector<float> createEmbedding(const Eigen::MatrixXd& mel) {
            if (mel.size() == 0) return {};
            int features = static_cast<int>(mel.rows());
            int frames = std::max(1, static_cast<int>(mel.cols()));

            Eigen::VectorXd mean = mel.rowwise().mean();
            Eigen::MatrixXd centered = mel.colwise() - mean;
            Eigen::VectorXd var = ((centered.array().square().rowwise().sum()) / frames).matrix();
            Eigen::VectorXd stddev = var.array().sqrt();
            for (int i = 0; i < stddev.size(); ++i)
                if (stddev(i) < 1e-9) stddev(i) = 1.0;

            int k = std::min(64, features);
            Eigen::MatrixXd cov = (centered * centered.transpose()) / std::max(1, frames);
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(cov);
            Eigen::VectorXd projections = Eigen::VectorXd::Zero(k);
            for (int i = 0; i < k; ++i) {
                int idx = features - 1 - i;
                if (idx < 0) break;
                Eigen::VectorXd vec = es.eigenvectors().col(idx);
                projections(i) = vec.dot(mean);
            }

            std::vector<float> emb;
            emb.reserve(features * 2 + k);
            for (int i = 0; i < features; ++i) emb.push_back(static_cast<float>(mean(i)));
            for (int i = 0; i < features; ++i) emb.push_back(static_cast<float>(stddev(i)));
            for (int i = 0; i < k; ++i) emb.push_back(static_cast<float>(projections(i)));

            double norm = 0.0;
            for (auto v : emb) norm += (double)v * (double)v;
            norm = std::sqrt(norm);
            if (norm > 1e-12)
                for (auto& v : emb) v = static_cast<float>((double)v / norm);

            return emb;
        }

        static double cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
            if (a.empty() || b.empty() || a.size() != b.size()) return -2.0;
            double dot = 0.0, na = 0.0, nb = 0.0;
            for (size_t i = 0; i < a.size(); ++i) {
                dot += (double)a[i] * (double)b[i];
                na += (double)a[i] * a[i];
                nb += (double)b[i] * b[i];
            }
            if (na <= 0 || nb <= 0) return -2.0;
            return dot / (std::sqrt(na) * std::sqrt(nb));
        }
    };

    // ==================== ONNXEmbedding (修正版) ====================
    class ONNXEmbedding {
    public:
        ONNXEmbedding(const std::string& modelPath, const Config& config)
            : initialized_(false)
        {
            try {
                env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "SpeakerID");
                Ort::SessionOptions sessionOptions;
                sessionOptions.SetIntraOpNumThreads(4);
                sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

                if (!modelPath.empty()) {
                    try {
                        OrtCUDAProviderOptions cuda_options;
                        cuda_options.device_id = config.cudaDeviceId;
                        sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
                        std::cout << "[SpeakerID] Using CUDA device " << config.cudaDeviceId << std::endl;
                    }
                    catch (...) {
                        std::cout << "[SpeakerID] CUDA not available, using CPU" << std::endl;
                    }
                }

#ifdef _WIN32
                std::wstring wModelPath(modelPath.begin(), modelPath.end());
                session_ = std::make_unique<Ort::Session>(*env_, wModelPath.c_str(), sessionOptions);
#else
                session_ = std::make_unique<Ort::Session>(*env_, modelPath.c_str(), sessionOptions);
#endif

                Ort::AllocatorWithDefaultOptions allocator;
                inputName_ = session_->GetInputNameAllocated(0, allocator).get();
                outputName_ = session_->GetOutputNameAllocated(0, allocator).get();

                auto inputTypeInfo = session_->GetInputTypeInfo(0);
                auto tensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
                inputShape_ = tensorInfo.GetShape();

                std::cout << "[SpeakerID] ONNX model loaded: " << modelPath << std::endl;
                std::cout << "[SpeakerID] Input: " << inputName_ << " shape=[";
                for (size_t i = 0; i < inputShape_.size(); ++i) {
                    std::cout << inputShape_[i];
                    if (i < inputShape_.size() - 1) std::cout << ", ";
                }
                std::cout << "]" << std::endl;
                std::cout << "[SpeakerID] Output: " << outputName_ << std::endl;

                initialized_ = true;
            }
            catch (const std::exception& ex) {
                std::cerr << "[SpeakerID] ONNX init failed: " << ex.what() << std::endl;
                initialized_ = false;
            }
        }

        bool isInitialized() const { return initialized_; }

        std::vector<float> extractEmbedding(const Eigen::MatrixXd& melSpec) {
            if (!initialized_) return {};
            if (melSpec.size() == 0) return {};

            int nMels = static_cast<int>(melSpec.rows());
            int frames = static_cast<int>(melSpec.cols());

            // 入力形式: [B, T, 80] = [1, frames, 80]
            std::vector<int64_t> shape = { 1, static_cast<int64_t>(frames), static_cast<int64_t>(nMels) };

            std::vector<float> inputData;
            inputData.reserve(frames * nMels);

            // [T, F] の順でデータを配置
            for (int t = 0; t < frames; ++t) {
                for (int m = 0; m < nMels; ++m) {
                    inputData.push_back(static_cast<float>(melSpec(m, t)));
                }
            }

            // デバッグ出力
            static int callCount = 0;
            callCount++;
            if (callCount <= 5 || callCount % 20 == 0) {
                float minVal = *std::min_element(inputData.begin(), inputData.end());
                float maxVal = *std::max_element(inputData.begin(), inputData.end());
                double mean = 0.0;
                for (auto v : inputData) mean += v;
                mean /= inputData.size();

                double stddev = 0.0;
                for (auto v : inputData) stddev += (v - mean) * (v - mean);
                stddev = std::sqrt(stddev / inputData.size());

                std::cout << "[ONNX #" << callCount << "] Input: ["
                    << shape[0] << "," << shape[1] << "," << shape[2] << "]"
                    << " range=[" << std::fixed << std::setprecision(3)
                    << minVal << "," << maxVal << "]"
                    << " mean=" << mean << " std=" << stddev << std::endl;
            }

            try {
                Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
                Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
                    mem, inputData.data(), inputData.size(), shape.data(), shape.size());

                const char* inNames[] = { inputName_.c_str() };
                const char* outNames[] = { outputName_.c_str() };

                auto outs = session_->Run(Ort::RunOptions{ nullptr },
                    inNames, &inputTensor, 1, outNames, 1);

                float* data = outs[0].GetTensorMutableData<float>();
                size_t size = outs[0].GetTensorTypeAndShapeInfo().GetElementCount();

                std::vector<float> embedding(data, data + size);

                if (callCount <= 5 || callCount % 20 == 0) {
                    float embMin = *std::min_element(embedding.begin(), embedding.end());
                    float embMax = *std::max_element(embedding.begin(), embedding.end());
                    double embMean = 0.0;
                    for (auto v : embedding) embMean += v;
                    embMean /= size;

                    std::cout << "[ONNX #" << callCount << "] Output: size=" << size
                        << " range=[" << std::fixed << std::setprecision(3)
                        << embMin << "," << embMax << "] mean=" << embMean << std::endl;
                }

                return embedding;

            }
            catch (const Ort::Exception& e) {
                std::cerr << "[ONNX Error] " << e.what() << std::endl;
                return {};
            }
        }

    private:
        std::unique_ptr<Ort::Env> env_;
        std::unique_ptr<Ort::Session> session_;
        std::string inputName_;
        std::string outputName_;
        std::vector<int64_t> inputShape_;
        bool initialized_;
    };

    // ==================== SpeakerModel ====================
    struct SpeakerModel {
        std::string name;
        std::vector<float> embedding;

        bool saveToFile(const fs::path& filename) const {
            std::ofstream ofs(filename, std::ios::binary);
            if (!ofs) return false;
            uint32_t nameLen = static_cast<uint32_t>(name.size());
            ofs.write(reinterpret_cast<const char*>(&nameLen), sizeof(nameLen));
            ofs.write(name.data(), nameLen);
            uint32_t embSize = static_cast<uint32_t>(embedding.size());
            ofs.write(reinterpret_cast<const char*>(&embSize), sizeof(embSize));
            ofs.write(reinterpret_cast<const char*>(embedding.data()), embSize * sizeof(float));
            return ofs.good();
        }

        bool loadFromFile(const fs::path& filename) {
            std::ifstream ifs(filename, std::ios::binary);
            if (!ifs) return false;
            uint32_t nameLen = 0;
            ifs.read(reinterpret_cast<char*>(&nameLen), sizeof(nameLen));
            name.resize(nameLen);
            ifs.read(&name[0], nameLen);
            uint32_t embSize = 0;
            ifs.read(reinterpret_cast<char*>(&embSize), sizeof(embSize));
            embedding.resize(embSize);
            ifs.read(reinterpret_cast<char*>(embedding.data()), embSize * sizeof(float));
            return ifs.good();
        }
    };

    // ==================== SpeakerIdentifier (改善版) ====================
    class SpeakerIdentifier {
    public:
        SpeakerIdentifier(const std::string& modelDir,
            const std::string& onnxModelPath = "",
            const Config& config = Config())
            : modelDir_(modelDir), config_(config)
        {
            melExtractor_ = std::make_unique<MelSpectrogramExtractor>(config_);

            if (!onnxModelPath.empty()) {
                onnxModel_ = std::make_unique<ONNXEmbedding>(onnxModelPath, config_);
                useONNX_ = onnxModel_->isInitialized();
                if (useONNX_) {
                    std::cout << "[SpeakerID] Using ONNX embeddings." << std::endl;
                }
                else {
                    std::cerr << "[SpeakerID] ONNX failed; using statistical embeddings." << std::endl;
                }
            }

            loadSpeakerModels();
        }

        bool enrollSpeaker(const std::string& name, const std::vector<float>& audioData) {
            if (name.empty() || audioData.empty()) return false;

            // 音声長チェック
            double duration = (double)audioData.size() / config_.sampleRate;
            if (duration < config_.minAudioSeconds) {
                std::cerr << "[Warning] Audio too short: " << duration
                    << "s (minimum " << config_.minAudioSeconds << "s)" << std::endl;
            }

            auto trimmed = trimSilence(audioData);
            if ((int)trimmed.size() < config_.sampleRate / 2) {
                if ((int)audioData.size() < config_.sampleRate / 2) return false;
                trimmed = audioData;
            }

            auto mel = melExtractor_->extract(trimmed);
            auto emb = computeEmbeddingByChunks(mel);
            if (emb.empty()) return false;

            l2normalize(emb);

            // ✅ 既存の話者モデルがあれば統合（Centroid Update）
            SpeakerModel m;
            m.name = name;
            m.embedding = emb;

            if (!fs::exists(modelDir_)) fs::create_directories(modelDir_);
            fs::path p = fs::path(modelDir_) / fs::u8path(name + ".bin");

            if (fs::exists(p)) {
                SpeakerModel existing;
                if (existing.loadFromFile(p)) {
                    if (existing.embedding.size() == emb.size()) {
                        std::cout << "[SpeakerID] Updating existing model for: " << name << std::endl;
                        for (size_t i = 0; i < emb.size(); ++i) {
                            m.embedding[i] += existing.embedding[i];
                        }
                        l2normalize(m.embedding);
                    }
                }
            }

            bool ok = m.saveToFile(p);
            if (ok) {
                // メモリ上のリストも更新
                bool found = false;
                for (auto& s : speakers_) {
                    if (s.name == name) {
                        s = m;
                        found = true;
                        break;
                    }
                }
                if (!found) speakers_.push_back(m);

                std::cout << "[SpeakerID] Enrolled: " << name
                    << " (embedding size: " << emb.size() << ")" << std::endl;
            }
            return ok;
        }

        IdentificationResult identify(const std::vector<float>& audioData) {
            IdentificationResult result;
            result.speakerName = "Unknown";
            result.confidence = -2.0;
            result.isUnknown = true;

            if (audioData.empty() || speakers_.empty()) return result;

            // 音声長チェック
            double duration = (double)audioData.size() / config_.sampleRate;
            if (duration < 0.5) {
                std::cerr << "[Warning] Audio too short for identification: "
                    << duration << "s" << std::endl;
                return result;
            }

            auto trimmed = trimSilence(audioData);
            if ((int)trimmed.size() < config_.sampleRate / 2) {
                if ((int)audioData.size() < config_.sampleRate / 2) return result;
                trimmed = audioData;
            }

            auto mel = melExtractor_->extract(trimmed);
            auto testEmb = computeEmbeddingByChunks(mel);
            if (testEmb.empty()) return result;
            l2normalize(testEmb);

            // 埋め込みの品質チェック
            double embStd = 0.0;
            double embMean = 0.0;
            for (auto v : testEmb) embMean += v;
            embMean /= testEmb.size();
            for (auto v : testEmb) embStd += (v - embMean) * (v - embMean);
            embStd = std::sqrt(embStd / testEmb.size());

            std::cout << "[Embedding] size=" << testEmb.size()
                << " std=" << std::fixed << std::setprecision(4) << embStd << std::endl;

            double maxSim = -2.0;
            std::string best = "Unknown";

            std::cout << "[Similarity Scores]" << std::endl;
            for (const auto& s : speakers_) {
                double sim = StatisticalEmbedding::cosineSimilarity(testEmb, s.embedding);
                result.allScores.push_back({ s.name, sim });
                std::cout << "  " << s.name << ": " << std::fixed
                    << std::setprecision(4) << sim << std::endl;
                if (sim > maxSim) { maxSim = sim; best = s.name; }
            }

            std::sort(result.allScores.begin(), result.allScores.end(),
                [](auto& a, auto& b) { return a.second > b.second; });

            result.confidence = maxSim;
            result.speakerName = best;
            result.isUnknown = (maxSim < config_.minConfidence);
            if (result.isUnknown) result.speakerName = "Unknown";

            return result;
        }

        std::vector<std::string> getSpeakerNames() const {
            std::vector<std::string> names;
            names.reserve(speakers_.size());
            for (auto& s : speakers_) names.push_back(s.name);
            return names;
        }

        size_t getSpeakerCount() const { return speakers_.size(); }

        bool removeSpeaker(const std::string& name) {
            auto it = std::find_if(speakers_.begin(), speakers_.end(),
                [&](const SpeakerModel& m) { return m.name == name; });
            if (it == speakers_.end()) return false;
            fs::path p = fs::path(modelDir_) / fs::u8path(name + ".bin");
            try {
                if (fs::exists(p)) fs::remove(p);
            }
            catch (...) {}
            speakers_.erase(it);
            return true;
        }

    public:
        std::string modelDir_;
        Config config_;
        std::unique_ptr<MelSpectrogramExtractor> melExtractor_;
        std::unique_ptr<ONNXEmbedding> onnxModel_;
        bool useONNX_ = false;
        std::vector<SpeakerModel> speakers_;

        static void l2normalize(std::vector<float>& v) {
            double s = 0.0;
            for (auto x : v) s += (double)x * (double)x;
            s = std::sqrt(s);
            if (s < 1e-12) return;
            for (auto& x : v) x = static_cast<float>((double)x / s);
        }

        static double calculateZCR(const float* data, size_t n) {
            int zero_crossings = 0;
            for (size_t i = 1; i < n; ++i) {
                if ((data[i - 1] >= 0 && data[i] < 0) ||
                    (data[i - 1] < 0 && data[i] >= 0)) {
                    zero_crossings++;
                }
            }
            return n ? double(zero_crossings) / double(n) : 0.0;
        }

        std::vector<float> trimSilence(const std::vector<float>& audio) const {
            const int win = 400;
            const double rmsThresh = 0.005; // 1e-4 -> 0.005 (より厳しく)
            const double zcrMin = 0.02;     // 足音などを除外
            const double zcrMax = 0.6;      // 高周波ノイズを除外

            int n = (int)audio.size();
            if (n == 0) return {};

            int start = 0;
            int end = n - 1;

            // 前方から探索
            for (int i = 0; i < n; i += win) {
                int jend = std::min(n, i + win);
                int len = jend - i;
                if (len <= 0) break;

                double sum = 0.0;
                for (int j = i; j < jend; ++j) sum += audio[j] * audio[j];
                double rms = std::sqrt(sum / len);
                double zcr = calculateZCR(&audio[i], len);

                if (rms > rmsThresh && zcr > zcrMin && zcr < zcrMax) {
                    start = std::max(0, i - win);
                    break;
                }
            }

            // 後方から探索
            for (int i = n - win; i >= 0; i -= win) {
                int jst = i;
                int jend = std::min(n, i + win);
                int len = jend - jst;
                if (len <= 0) break;

                double sum = 0.0;
                for (int j = jst; j < jend; ++j) sum += audio[j] * audio[j];
                double rms = std::sqrt(sum / len);
                double zcr = calculateZCR(&audio[jst], len);

                if (rms > rmsThresh && zcr > zcrMin && zcr < zcrMax) {
                    end = std::min(n - 1, i + win * 2);
                    break;
                }
            }

            int margin = static_cast<int>(0.3 * config_.sampleRate);
            start = std::max(0, start - margin);
            end = std::min(n - 1, end + margin);
            if (end <= start) return {};

            std::vector<float> out;
            out.reserve(end - start + 1);
            for (int i = start; i <= end; ++i) out.push_back(audio[i]);
            return out;
        }

        // ✅ 改善版: チャンク処理
        std::vector<float> computeEmbeddingByChunks(const Eigen::MatrixXd& melSpec) {
            int frames = static_cast<int>(melSpec.cols());
            int nMels = static_cast<int>(melSpec.rows());

            if (frames <= 0) return {};

            // 短い音声は直接処理
            if (!useONNX_ || frames <= 300) {
                auto emb = useONNX_ ? onnxModel_->extractEmbedding(melSpec)
                    : StatisticalEmbedding::createEmbedding(melSpec);
                if (!emb.empty()) l2normalize(emb);
                return emb;
            }

            // ✅ 長い音声: オーバーラップウィンドウ処理
            const int windowSize = 300;
            const int stride = 150;  // 50% オーバーラップ

            std::vector<std::vector<float>> embeddings;

            // スライディングウィンドウ
            for (int start = 0; start <= frames - windowSize; start += stride) {
                Eigen::MatrixXd window = melSpec.block(0, start, nMels, windowSize);
                auto emb = onnxModel_->extractEmbedding(window);

                if (!emb.empty()) {
                    // ✅ 各チャンクを正規化してから保存
                    l2normalize(emb);
                    embeddings.push_back(emb);
                }
            }

            // 最後のウィンドウ（末尾から）
            int lastStart = frames - windowSize;
            if (lastStart > 0 && embeddings.empty()) {
                Eigen::MatrixXd window = melSpec.block(0, lastStart, nMels, windowSize);
                auto emb = onnxModel_->extractEmbedding(window);
                if (!emb.empty()) {
                    l2normalize(emb);
                    embeddings.push_back(emb);
                }
            }

            if (embeddings.empty()) return {};

            // ✅ 平均埋め込みを計算
            size_t embSize = embeddings[0].size();
            std::vector<float> avgEmb(embSize, 0.0f);

            for (const auto& emb : embeddings) {
                for (size_t i = 0; i < embSize; ++i) {
                    avgEmb[i] += emb[i];
                }
            }

            for (size_t i = 0; i < embSize; ++i) {
                avgEmb[i] /= embeddings.size();
            }

            // ✅ 最後にもう一度L2正規化
            l2normalize(avgEmb);

            std::cout << "[SpeakerID] Processed " << embeddings.size()
                << " windows for " << frames << " frames" << std::endl;

            return avgEmb;
        }

        void loadSpeakerModels() {
            speakers_.clear();
            try {
                if (!fs::exists(modelDir_)) return;
                for (auto& p : fs::directory_iterator(modelDir_)) {
                    if (!p.is_regular_file()) continue;
                    auto path = p.path();
                    if (path.extension() == ".bin") {
                        SpeakerModel m;
                        if (m.loadFromFile(path)) {
                            l2normalize(m.embedding);
                            speakers_.push_back(std::move(m));
                            std::cout << "[SpeakerID] Loaded: " << m.name
                                << " (embedding: " << m.embedding.size() << ")" << std::endl;
                        }
                    }
                }
            }
            catch (const std::exception& ex) {
                std::cerr << "[SpeakerID] loadSpeakerModels error: " << ex.what() << std::endl;
            }
        }
    };


#include "SpeakerIdentifier.h"
    using namespace SpeakerID;

   

} // namespace SpeakerID


