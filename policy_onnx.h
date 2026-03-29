#ifndef ACTDOG_POLICY_ONNX_H_
#define ACTDOG_POLICY_ONNX_H_

#include <array>
#include <deque>
#include <memory>
#include <string>
#include <vector>

typedef struct mjModel_ mjModel;
typedef struct mjData_ mjData;

namespace actdog
{
  class ActPolicyOnnxRunner
  {
  public:
    ActPolicyOnnxRunner();
    ~ActPolicyOnnxRunner();

    ActPolicyOnnxRunner(const ActPolicyOnnxRunner &) = delete;
    ActPolicyOnnxRunner &operator=(const ActPolicyOnnxRunner &) = delete;

    // Load the exported ONNX file and its sidecar deployment metadata JSON.
    bool Load(const std::string &onnx_path, const std::string &metadata_path = "");

    // Clear the cached ACT action chunk. Call this whenever a new episode starts.
    void Reset();

    // Predict one action.
    //
    // This function is designed to be called from a faster control loop.
    // It internally:
    // 1. holds the current action until the next 1/fps boundary
    // 2. runs ONNX only when a new ACT chunk is needed
    // 3. consumes the chunk one action at a time, matching ACT's select_action behavior
    bool Predict(
        double sim_time_s,
        const std::vector<float> &state,
        const std::vector<unsigned char> &rgb_hwc,
        int image_width,
        int image_height,
        std::array<float, 3> &action_out);

    const std::string &GetLastError() const { return last_error_; }
    int GetExpectedWidth() const { return image_width_; }
    int GetExpectedHeight() const { return image_height_; }
    float GetDeploymentFps() const { return deployment_fps_; }

  private:
    bool LoadMetadata(const std::string &metadata_path);
    bool RunModel(const std::vector<float> &state,
                  const std::vector<unsigned char> &rgb_hwc,
                  int image_width,
                  int image_height);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    std::deque<std::array<float, 3>> action_queue_;
    std::array<float, 3> current_action_ = {0.0f, 0.0f, 0.0f};
    bool has_current_action_ = false;
    double next_action_time_s_ = 0.0;

    int image_width_ = 160;
    int image_height_ = 120;
    int state_dim_ = 3;
    int action_dim_ = 3;
    int n_action_steps_ = 1;
    float deployment_fps_ = 20.0f;
    std::string last_error_;
  };

  // Convenience function: static singleton runner keyed by paths.
  bool InferActPolicyAction(
      const std::string &onnx_path,
      const std::string &metadata_path,
      double sim_time_s,
      const std::vector<float> &state,
      const std::vector<unsigned char> &rgb_hwc,
      int image_width,
      int image_height,
      std::array<float, 3> &action_out,
      std::string *error = nullptr);

  // Project-oriented helper: render `camera1` from the current MuJoCo state and
  // directly infer one action. The policy checkpoint currently consumes only
  // state + RGB, so radar is not passed here.
  bool InferActPolicyActionFromMujoco(
      const std::string &onnx_path,
      const std::string &metadata_path,
      double sim_time_s,
      const mjModel *model,
      mjData *data,
      const std::vector<float> &state,
      std::array<float, 3> &action_out,
      const std::string &camera_name = "camera1",
      std::string *error = nullptr);

  // Clear the cached singleton runners used by the convenience helpers above.
  void ResetActPolicyInferenceCache();
}

#endif
