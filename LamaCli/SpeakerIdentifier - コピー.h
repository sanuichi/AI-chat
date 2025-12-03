// SpeakerIdentifier_rewrite.cpp
// 完全リファクタ版: SpeakerIdentifier (ヘッダ + 実装を1ファイルに統合)
// 目的: 高精度な話者識別 (ONNX ResNet 埋め込み + チャンク平均化 + 改良前処理)
// 依存: Eigen, FFTW3, ONNX Runtime, C++17 filesystem

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

#include <Eigen/Dense>
#include <fftw3.h>
#include <onnxruntime_cxx_api.h>

#define M_PI 3.14159265358979323846
namespace SpeakerID {

    namespace fs = std::filesystem;

    // -------------------- 設定構造体 --------------------
    struct Config {
        int sampleRate = 16000;      // サンプリングレート
        int nMels = 80;              // メルフィルタバンク数
        int fftSize = 1024;          // FFTサイズ
        int hopSize = 160;           // ホップサイズ
        int cudaDeviceId = 0;        // GPU ID for ONNX
        bool useFP16 = false;        // ONNX FP16
        bool useCPUFallback = true;  // CUDA失敗時にCPUでフォールバック
        double minConfidence = 0.70; // 識別閾値: 0.7 を推奨（変更可）
        int chunkFrames = 80;        // 埋め込みを算出するフレーム長（メルフレーム単位, 80 ≒ 0.8s@160 hop）
    };

    // -------------------- 識別結果 --------------------
    struct IdentificationResult {
        std::string speakerName;              // 識別された話者名 (Unknown の場合は "Unknown")
        double confidence;                    // 信頼度 (-1..1 の cosine 類似値)
        bool isUnknown;                       // 未知の話者か
        std::vector<std::pair<std::string, double>> allScores; // ソート済みの全話者のスコア (降順)

        std::string getConfidenceLevel() const {
            if (confidence >= 0.85) return "very_high";
            if (confidence >= 0.75) return "high";
            if (confidence >= 0.65) return "medium";
            if (confidence >= 0.55) return "low";
            return "very_low";
        }

        std::string getConfidenceDescription() const {
            // 単純にラベル戻す
            return getConfidenceLevel();
        }
    };

    // -------------------- MelSpectrogramExtractor --------------------
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

        // audio_in: float [-1..1] モノラル
        // 戻り値: (nMels x frames) の行列
        Eigen::MatrixXd extract(const std::vector<float>& audio_in) {
            if (audio_in.empty()) return Eigen::MatrixXd::Zero(nMels_, 1);

            // 正規化 (peak)
            std::vector<float> audio = audio_in;
            float maxAmp = 0.0f;
            for (auto s : audio) maxAmp = std::max(maxAmp, std::fabs(s));
            if (maxAmp > 1e-6f) for (auto& s : audio) s /= maxAmp;

            if ((int)audio.size() < fftSize_) {
                // パディング
                std::vector<float> tmp(fftSize_, 0.0f);
                std::copy(audio.begin(), audio.end(), tmp.begin());
                audio.swap(tmp);
            }

            int numFrames = static_cast<int>((audio.size() - fftSize_) / hopSize_) + 1;
            if (numFrames <= 0) numFrames = 1;

            Eigen::MatrixXd melSpec(nMels_, numFrames);

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
                // log1p (natural log) を利用: log(mel + eps)
                mel = (mel.array() + 1e-6).log();
                melSpec.col(i) = mel;
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
                bins[i] = std::min(nfft - 1, std::max(0, static_cast<int>(std::floor((fftSize_ + 1) * hz / sampleRate_))));
            }
            for (int i = 0; i < nMels_; ++i) {
                int left = bins[i], center = bins[i + 1], right = bins[i + 2];
                if (center == left) center = left + 1;
                if (right == center) right = center + 1;
                center = std::min(center, nfft - 1);
                right = std::min(right, nfft - 1);
                for (int j = left; j < center; ++j) if (j >= 0 && j < nfft) fb(i, j) = (j - left) / double(center - left);
                for (int j = center; j < right; ++j) if (j >= 0 && j < nfft) fb(i, j) = (right - j) / double(right - center);
            }
            return fb;
        }
    };

    // -------------------- StatisticalEmbedding (フォールバック) --------------------
    class StatisticalEmbedding {
    public:
        // メルスペクトログラム (nMels x frames) から統計的embeddingを作成
        static std::vector<float> createEmbedding(const Eigen::MatrixXd& mel) {
            if (mel.size() == 0) return {};
            int features = static_cast<int>(mel.rows());
            int frames = std::max(1, static_cast<int>(mel.cols()));

            Eigen::VectorXd mean = mel.rowwise().mean();
            Eigen::MatrixXd centered = mel.colwise() - mean;
            Eigen::VectorXd var = ((centered.array().square().rowwise().sum()) / frames).matrix();
            Eigen::VectorXd stddev = var.array().sqrt();
            for (int i = 0; i < stddev.size(); ++i) if (stddev(i) < 1e-9) stddev(i) = 1.0;

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

            // L2 正規化
            double norm = 0.0;
            for (auto v : emb) norm += (double)v * (double)v;
            norm = std::sqrt(norm);
            if (norm > 1e-12) for (auto& v : emb) v = static_cast<float>((double)v / norm);
            return emb;
        }

        static double cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
            if (a.empty() || b.empty() || a.size() != b.size()) return -2.0;
            double dot = 0.0, na = 0.0, nb = 0.0;
            for (size_t i = 0; i < a.size(); ++i) { dot += (double)a[i] * (double)b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i]; }
            if (na <= 0 || nb <= 0) return -2.0;
            return dot / (std::sqrt(na) * std::sqrt(nb));
        }
    };

    // -------------------- ONNXEmbedding --------------------
    class ONNXEmbedding {
    public:
        ONNXEmbedding(const std::string& modelPath, const Config& config)
            : initialized_(false), useCPU_(false)
        {
            try {
                env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_ERROR, "SpeakerID");
                Ort::SessionOptions sessionOptions;
                sessionOptions.SetIntraOpNumThreads(4);
                sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

                // Try append CUDA provider if requested
                if (!modelPath.empty()) {
                    try {
                        OrtCUDAProviderOptions cuda_options;
                        cuda_options.device_id = config.cudaDeviceId;
                        sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
                    }
                    catch (...) {
                        throw;
                    }
                }

                // Load model
#ifdef _WIN32
                std::wstring wModelPath(modelPath.begin(), modelPath.end());
                session_ = std::make_unique<Ort::Session>(*env_, wModelPath.c_str(), sessionOptions);
#else
                session_ = std::make_unique<Ort::Session>(*env_, modelPath.c_str(), sessionOptions);
#endif

                Ort::AllocatorWithDefaultOptions allocator;
                inputName_ = session_->GetInputNameAllocated(0, allocator).get();
                outputName_ = session_->GetOutputNameAllocated(0, allocator).get();

                initialized_ = true;
                std::cout << "[SpeakerID] ONNX model loaded: " << modelPath << std::endl;
            }
            catch (const std::exception& ex) {
                std::cerr << "[SpeakerID] ONNX init failed: " << ex.what() << std::endl;
                initialized_ = false;
            }
            catch (...) {
                std::cerr << "[SpeakerID] ONNX init failed: unknown error" << std::endl;
                initialized_ = false;
            }
        }

        bool isInitialized() const { return initialized_; }

        // melSpec: nMels x frames
        std::vector<float> extractEmbedding(const Eigen::MatrixXd& melSpec) {
            if (!initialized_) return {};
            if (melSpec.size() == 0) return {};

            // ONNX model commonly expects (1, frames, nMels) or (1, nMels, frames). Here assume (1, frames, nMels).
            int frames = static_cast<int>(melSpec.cols());
            int nMels = static_cast<int>(melSpec.rows());

            // transpose to (frames x nMels)
            Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> inputT = melSpec.transpose().cast<float>();
            std::vector<int64_t> shape = { 1, frames, nMels };
            std::vector<float> inputData(inputT.data(), inputT.data() + inputT.size());

            Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value inputTensor = Ort::Value::CreateTensor<float>(mem, inputData.data(), inputData.size(), shape.data(), shape.size());

            const char* inNames[] = { inputName_.c_str() };
            const char* outNames[] = { outputName_.c_str() };
            auto outs = session_->Run(Ort::RunOptions{ nullptr }, inNames, &inputTensor, 1, outNames, 1);

            float* data = outs[0].GetTensorMutableData<float>();
            size_t size = outs[0].GetTensorTypeAndShapeInfo().GetElementCount();
            return std::vector<float>(data, data + size);
        }

    private:
        std::unique_ptr<Ort::Env> env_;
        std::unique_ptr<Ort::Session> session_;
        std::string inputName_;
        std::string outputName_;
        bool initialized_;
        bool useCPU_;
    };

    // -------------------- SpeakerModel --------------------
    struct SpeakerModel {
        std::string name;
        std::vector<float> embedding; // L2 正規化済

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

    // -------------------- SpeakerIdentifier --------------------
    class SpeakerIdentifier {
    public:
        SpeakerIdentifier(const std::string& modelDir,
            const std::string& onnxModelPath = "",
            const Config& config = Config())
            : modelDir_(modelDir), config_(config)
        {
            melExtractor_ = std::make_unique<MelSpectrogramExtractor>(config_);

            // Try load ONNX if available
            if (!onnxModelPath.empty()) {
                onnxModel_ = std::make_unique<ONNXEmbedding>(onnxModelPath, config_);
                useONNX_ = onnxModel_->isInitialized();
                if (useONNX_) {
                    std::cout << "[SpeakerID] Using ONNX embeddings." << std::endl;
                }
                else {
                    std::cerr << "[SpeakerID] ONNX not available; falling back to statistical embeddings." << std::endl;
                }
            }

            // load existing speaker models
            loadSpeakerModels();
        }

        // 登録
        bool enrollSpeaker(const std::string& name, const std::vector<float>& audioData) {
            if (name.empty() || audioData.empty()) return false;

            auto trimmed = trimSilence(audioData);
            if ((int)trimmed.size() < config_.sampleRate / 2) {
                // 最低 0.5s を要求
                if ((int)audioData.size() < config_.sampleRate / 2) return false;
                trimmed = audioData;
            }

            auto mel = melExtractor_->extract(trimmed);
            auto emb = computeEmbeddingByChunks(mel);
            if (emb.empty()) return false;

            // L2 正規化
            l2normalize(emb);

            SpeakerModel m;
            m.name = name;
            m.embedding = emb;

            if (!fs::exists(modelDir_)) fs::create_directories(modelDir_);
            fs::path p = fs::path(modelDir_) / fs::u8path(name + ".bin");
            bool ok = m.saveToFile(p);
            if (ok) speakers_.push_back(m);
            return ok;
        }

        IdentificationResult identify(const std::vector<float>& audioData) {
            IdentificationResult result;
            result.speakerName = "Unknown";
            result.confidence = -2.0;
            result.isUnknown = true;

            if (audioData.empty() || speakers_.empty()) return result;

            auto trimmed = trimSilence(audioData);
            if ((int)trimmed.size() < config_.sampleRate / 2) {
                if ((int)audioData.size() < config_.sampleRate / 2) return result;
                trimmed = audioData;
            }

            auto mel = melExtractor_->extract(trimmed);
            auto testEmb = computeEmbeddingByChunks(mel);
            if (testEmb.empty()) return result;
            l2normalize(testEmb);

            double maxSim = -2.0;
            std::string best = "Unknown";
            for (const auto& s : speakers_) {
                double sim = StatisticalEmbedding::cosineSimilarity(testEmb, s.embedding);
                result.allScores.push_back({ s.name, sim });
                if (sim > maxSim) { maxSim = sim; best = s.name; }
            }

            std::sort(result.allScores.begin(), result.allScores.end(), [](auto& a, auto& b) { return a.second > b.second; });

            result.confidence = maxSim;
            result.speakerName = best;
            result.isUnknown = (maxSim < config_.minConfidence);
            if (result.isUnknown) result.speakerName = "Unknown";
            return result;
        }

        std::vector<std::string> getSpeakerNames() const {
            std::vector<std::string> names; names.reserve(speakers_.size());
            for (auto& s : speakers_) names.push_back(s.name);
            return names;
        }

        size_t getSpeakerCount() const { return speakers_.size(); }

        bool removeSpeaker(const std::string& name) {
            auto it = std::find_if(speakers_.begin(), speakers_.end(), [&](const SpeakerModel& m) { return m.name == name; });
            if (it == speakers_.end()) return false;
            fs::path p = fs::path(modelDir_) / fs::u8path(name + ".bin");
            try { if (fs::exists(p)) fs::remove(p); }
            catch (...) {}
            speakers_.erase(it);
            return true;
        }

    private:
        std::string modelDir_;
        Config config_;
        std::unique_ptr<MelSpectrogramExtractor> melExtractor_;
        std::unique_ptr<ONNXEmbedding> onnxModel_;
        bool useONNX_ = false;
        std::vector<SpeakerModel> speakers_;

        // -------------------- ユーティリティ --------------------
        static void l2normalize(std::vector<float>& v) {
            double s = 0.0; for (auto x : v) s += (double)x * (double)x; s = std::sqrt(s); if (s < 1e-12) return; for (auto& x : v) x = (float)((double)x / s);
        }

        // 無音トリミング（前後）と、短い音声の保護マージンを付与
        std::vector<float> trimSilence(const std::vector<float>& audio) const {
            // 単純なエネルギー閾値によるトリミング
            const int win = 400; // 25ms @16k
            const double thresh = 1e-4; // RMS threshold
            int n = (int)audio.size();
            if (n == 0) return {};

            int start = 0; int end = n - 1;
            // 前方
            for (int i = 0; i < n; i += win) {
                int jend = std::min(n, i + win);
                double sum = 0.0;
                for (int j = i; j < jend; ++j) sum += audio[j] * audio[j];
                double rms = std::sqrt(sum / (jend - i));
                if (rms > thresh) { start = std::max(0, i - win); break; }
            }
            // 後方
            for (int i = n - win; i >= 0; i -= win) {
                int jst = i;
                int jend = std::min(n, i + win);
                double sum = 0.0;
                for (int j = jst; j < jend; ++j) sum += audio[j] * audio[j];
                double rms = std::sqrt(sum / (jend - jst));
                if (rms > thresh) { end = std::min(n - 1, i + win * 2); break; }
            }

            // マージン
            int margin = static_cast<int>(0.3 * config_.sampleRate); // 0.3 sec margin
            start = std::max(0, start - margin);
            end = std::min(n - 1, end + margin);
            if (end <= start) return {};
            std::vector<float> out;
            out.reserve(end - start + 1);
            for (int i = start; i <= end; ++i) out.push_back(audio[i]);
            return out;
        }

        // melSpec をチャンクに分け、各チャンクで embedding を取り平均化する
        std::vector<float> computeEmbeddingByChunks(const Eigen::MatrixXd& melSpec) {
            int frames = static_cast<int>(melSpec.cols());
            int nMels = static_cast<int>(melSpec.rows());
            if (frames <= 0) return {};

            int chunk = std::max(1, config_.chunkFrames);
            std::vector<std::vector<float>> chunkEmbeds;

            for (int start = 0; start < frames; start += chunk) {
                int end = std::min(frames, start + chunk);
                Eigen::MatrixXd part = melSpec.block(0, start, nMels, end - start);
                std::vector<float> emb;
                if (useONNX_) emb = onnxModel_->extractEmbedding(part);
                if (emb.empty()) emb = StatisticalEmbedding::createEmbedding(part);
                if (!emb.empty()) chunkEmbeds.push_back(std::move(emb));
            }

            if (chunkEmbeds.empty()) return {};

            // サイズ整合: すべて同じ長さであることを期待。もし違うなら最小長に合わせる
            size_t minSize = chunkEmbeds[0].size();
            for (auto& e : chunkEmbeds) if (e.size() < minSize) minSize = e.size();
            if (minSize == 0) return {};

            std::vector<float> avg(minSize, 0.0f);
            for (auto& e : chunkEmbeds) {
                for (size_t i = 0; i < minSize; ++i) avg[i] += e[i];
            }
            for (size_t i = 0; i < minSize; ++i) avg[i] /= (float)chunkEmbeds.size();
            return avg;
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
                            // 正規化確認
                            l2normalize(m.embedding);
                            speakers_.push_back(std::move(m));
                        }
                    }
                }
            }
            catch (const std::exception& ex) {
                std::cerr << "[SpeakerID] loadSpeakerModels error: " << ex.what() << std::endl;
            }
        }
    };

} // namespace SpeakerID

// EOF
