#include "policy_onnx.h"

#include <algorithm>
#include <fstream>
#include <iostream>

#include "Self_mujoco_lib.h"
#include "json.hpp"

#ifndef ACTDOG_HAS_ONNXRUNTIME
#define ACTDOG_HAS_ONNXRUNTIME 0
#endif

#if ACTDOG_HAS_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace actdog
{
#if ACTDOG_HAS_ONNXRUNTIME
  struct ActPolicyOnnxRunner::Impl
  {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "actdog_policy_onnx"};
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    std::string input_name_state;
    std::string input_name_image;
    std::string output_name_action_chunk;
  };
#else
  struct ActPolicyOnnxRunner::Impl
  {
  };
#endif

  namespace
  {
    struct GlobalRunnerState
    {
      ActPolicyOnnxRunner runner;
      std::string loaded_onnx_path;
      std::string loaded_metadata_path;
    };

    std::string DefaultMetadataPathFromOnnx(const std::string &onnx_path)
    {
      const std::string suffix = ".onnx";
      if (onnx_path.size() >= suffix.size() &&
          onnx_path.compare(onnx_path.size() - suffix.size(), suffix.size(), suffix) == 0)
      {
        return onnx_path.substr(0, onnx_path.size() - suffix.size()) + ".json";
      }
      return onnx_path + ".json";
    }

    GlobalRunnerState &RawRunnerState()
    {
      static GlobalRunnerState state;
      return state;
    }

    GlobalRunnerState &MujocoRunnerState()
    {
      static GlobalRunnerState state;
      return state;
    }
  } // namespace

  ActPolicyOnnxRunner::ActPolicyOnnxRunner() : impl_(std::make_unique<Impl>())
  {
#if ACTDOG_HAS_ONNXRUNTIME
    impl_->session_options.SetIntraOpNumThreads(1);
    impl_->session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
#endif
  }

  ActPolicyOnnxRunner::~ActPolicyOnnxRunner() = default;

  bool ActPolicyOnnxRunner::LoadMetadata(const std::string &metadata_path)
  {
    std::ifstream file(metadata_path);
    if (!file.is_open())
    {
      last_error_ = "Failed to open metadata file: " + metadata_path;
      return false;
    }

    nlohmann::json j;
    file >> j;

    state_dim_ = j.value("state_dim", 3);
    action_dim_ = j.value("action_dim", 3);
    n_action_steps_ = j.value("n_action_steps", 1);
    deployment_fps_ = j.value("deployment_fps", 20.0f);

    if (j.contains("image_shape_chw") && j["image_shape_chw"].is_array() && j["image_shape_chw"].size() == 3)
    {
      image_height_ = j["image_shape_chw"][1].get<int>();
      image_width_ = j["image_shape_chw"][2].get<int>();
    }

    if (j.contains("warning") && !j["warning"].is_null())
    {
      std::cout << "ActPolicyOnnxRunner metadata warning: " << j["warning"].get<std::string>() << std::endl;
    }

    return true;
  }

  bool ActPolicyOnnxRunner::Load(const std::string &onnx_path, const std::string &metadata_path)
  {
    Reset();

    const std::string resolved_metadata_path =
        metadata_path.empty() ? DefaultMetadataPathFromOnnx(onnx_path) : metadata_path;
    if (!LoadMetadata(resolved_metadata_path))
    {
      return false;
    }

#if ACTDOG_HAS_ONNXRUNTIME
    try
    {
      impl_->session = std::make_unique<Ort::Session>(impl_->env, onnx_path.c_str(), impl_->session_options);

      Ort::AllocatorWithDefaultOptions allocator;
      impl_->input_name_state = impl_->session->GetInputNameAllocated(0, allocator).get();
      impl_->input_name_image = impl_->session->GetInputNameAllocated(1, allocator).get();
      impl_->output_name_action_chunk = impl_->session->GetOutputNameAllocated(0, allocator).get();
    }
    catch (const std::exception &e)
    {
      last_error_ = std::string("Failed to load ONNX session: ") + e.what();
      return false;
    }
    return true;
#else
    (void)onnx_path;
    last_error_ =
        "ONNX Runtime support is not enabled in this build. Install onnxruntime and rebuild with CMake.";
    return false;
#endif
  }

  void ActPolicyOnnxRunner::Reset()
  {
    action_queue_.clear();
    current_action_ = {0.0f, 0.0f, 0.0f};
    has_current_action_ = false;
    next_action_time_s_ = 0.0;
    last_error_.clear();
  }

  bool ActPolicyOnnxRunner::RunModel(const std::vector<float> &state,
                                     const std::vector<unsigned char> &rgb_hwc,
                                     int image_width,
                                     int image_height)
  {
#if ACTDOG_HAS_ONNXRUNTIME
    if (!impl_->session)
    {
      last_error_ = "ONNX session is not loaded.";
      return false;
    }
    if (static_cast<int>(state.size()) != state_dim_)
    {
      last_error_ = "State dimension does not match the exported ONNX policy.";
      return false;
    }
    if (image_width != image_width_ || image_height != image_height_)
    {
      last_error_ = "Unexpected image size for policy deployment.";
      return false;
    }
    if (rgb_hwc.size() != static_cast<size_t>(image_width * image_height * 3))
    {
      last_error_ = "RGB buffer size does not match image dimensions.";
      return false;
    }
    std::vector<float> state_buffer(state.begin(), state.end());
    std::vector<float> image_buffer(static_cast<size_t>(image_width * image_height * 3), 0.0f);

    // Convert HWC uint8 RGB into CHW float32 as required by the exported ONNX model.
    for (int y = 0; y < image_height; ++y)
    {
      for (int x = 0; x < image_width; ++x)
      {
        const size_t src = static_cast<size_t>((y * image_width + x) * 3);
        const size_t dst_r = static_cast<size_t>(0 * image_height * image_width + y * image_width + x);
        const size_t dst_g = static_cast<size_t>(1 * image_height * image_width + y * image_width + x);
        const size_t dst_b = static_cast<size_t>(2 * image_height * image_width + y * image_width + x);
        image_buffer[dst_r] = static_cast<float>(rgb_hwc[src + 0]);
        image_buffer[dst_g] = static_cast<float>(rgb_hwc[src + 1]);
        image_buffer[dst_b] = static_cast<float>(rgb_hwc[src + 2]);
      }
    }

    const std::array<int64_t, 2> state_shape = {1, state_dim_};
    const std::array<int64_t, 4> image_shape = {1, 3, image_height_, image_width_};

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value state_tensor =
        Ort::Value::CreateTensor<float>(memory_info, state_buffer.data(), state_buffer.size(),
                                        state_shape.data(), state_shape.size());
    Ort::Value image_tensor =
        Ort::Value::CreateTensor<float>(memory_info, image_buffer.data(), image_buffer.size(),
                                        image_shape.data(), image_shape.size());

    const char *input_names[] = {impl_->input_name_state.c_str(), impl_->input_name_image.c_str()};
    Ort::Value input_tensors[] = {std::move(state_tensor), std::move(image_tensor)};
    const char *output_names[] = {impl_->output_name_action_chunk.c_str()};

    try
    {
      auto output_tensors = impl_->session->Run(
          Ort::RunOptions{nullptr},
          input_names,
          input_tensors,
          2,
          output_names,
          1);

      if (output_tensors.empty())
      {
        last_error_ = "ONNX inference returned no outputs.";
        return false;
      }

      Ort::Value &output = output_tensors.front();
      auto type_info = output.GetTensorTypeAndShapeInfo();
      std::vector<int64_t> shape = type_info.GetShape();
      if (shape.size() != 3 || shape[0] != 1 || shape[2] != action_dim_)
      {
        last_error_ = "Unexpected action_chunk output shape.";
        return false;
      }

      const float *actions = output.GetTensorData<float>();
      const int step_count = static_cast<int>(shape[1]);
      action_queue_.clear();
      action_queue_.resize(step_count);
      for (int step = 0; step < step_count; ++step)
      {
        for (int dim = 0; dim < action_dim_; ++dim)
        {
          action_queue_[step][dim] = actions[step * action_dim_ + dim];
        }
      }
      return true;
    }
    catch (const std::exception &e)
    {
      last_error_ = std::string("ONNX inference failed: ") + e.what();
      return false;
    }
#else
    (void)state;
    (void)rgb_hwc;
    (void)image_width;
    (void)image_height;
    last_error_ =
        "ONNX Runtime support is not enabled in this build. Install onnxruntime and rebuild with CMake.";
    return false;
#endif
  }

  bool ActPolicyOnnxRunner::Predict(
      double sim_time_s,
      const std::vector<float> &state,
      const std::vector<unsigned char> &rgb_hwc,
      int image_width,
      int image_height,
      std::array<float, 3> &action_out)
  {
    const double action_period_s = deployment_fps_ > 0.0f ? 1.0 / deployment_fps_ : 0.05;

    if (!has_current_action_ || sim_time_s + 1e-9 >= next_action_time_s_)
    {
      if (action_queue_.empty())
      {
        if (!RunModel(state, rgb_hwc, image_width, image_height))
        {
          return false;
        }
      }

      if (action_queue_.empty())
      {
        last_error_ = "Policy action queue is unexpectedly empty.";
        return false;
      }

      current_action_ = action_queue_.front();
      action_queue_.pop_front();
      has_current_action_ = true;
      next_action_time_s_ = sim_time_s + action_period_s;
    }

    action_out = current_action_;
    return true;
  }

  bool InferActPolicyAction(
      const std::string &onnx_path,
      const std::string &metadata_path,
      double sim_time_s,
      const std::vector<float> &state,
      const std::vector<unsigned char> &rgb_hwc,
      int image_width,
      int image_height,
      std::array<float, 3> &action_out,
      std::string *error)
  {
    GlobalRunnerState &cache = RawRunnerState();

    if (onnx_path != cache.loaded_onnx_path || metadata_path != cache.loaded_metadata_path)
    {
      if (!cache.runner.Load(onnx_path, metadata_path))
      {
        if (error)
        {
          *error = cache.runner.GetLastError();
        }
        return false;
      }
      cache.loaded_onnx_path = onnx_path;
      cache.loaded_metadata_path = metadata_path;
    }

    const bool ok = cache.runner.Predict(sim_time_s, state, rgb_hwc, image_width, image_height, action_out);
    if (!ok && error)
    {
      *error = cache.runner.GetLastError();
    }
    return ok;
  }

  bool InferActPolicyActionFromMujoco(
      const std::string &onnx_path,
      const std::string &metadata_path,
      double sim_time_s,
      const mjModel *model,
      mjData *data,
      const std::vector<float> &state,
      std::array<float, 3> &action_out,
      const std::string &camera_name,
      std::string *error)
  {
    GlobalRunnerState &cache = MujocoRunnerState();

    if (onnx_path != cache.loaded_onnx_path || metadata_path != cache.loaded_metadata_path)
    {
      if (!cache.runner.Load(onnx_path, metadata_path))
      {
        if (error)
        {
          *error = cache.runner.GetLastError();
        }
        return false;
      }
      cache.loaded_onnx_path = onnx_path;
      cache.loaded_metadata_path = metadata_path;
    }

    std::vector<unsigned char> rgb;
    if (!mujo::render_camera_rgb(
            model,
            data,
            camera_name,
            cache.runner.GetExpectedWidth(),
            cache.runner.GetExpectedHeight(),
            rgb,
            true))
    {
      if (error)
      {
        *error = "Failed to render policy camera: " + camera_name;
      }
      return false;
    }

    const bool ok = cache.runner.Predict(
        sim_time_s,
        state,
        rgb,
        cache.runner.GetExpectedWidth(),
        cache.runner.GetExpectedHeight(),
        action_out);
    if (!ok && error)
    {
      *error = cache.runner.GetLastError();
    }
    return ok;
  }

  void ResetActPolicyInferenceCache()
  {
    GlobalRunnerState &raw_cache = RawRunnerState();
    raw_cache.runner.Reset();
    raw_cache.loaded_onnx_path.clear();
    raw_cache.loaded_metadata_path.clear();

    GlobalRunnerState &mujoco_cache = MujocoRunnerState();
    mujoco_cache.runner.Reset();
    mujoco_cache.loaded_onnx_path.clear();
    mujoco_cache.loaded_metadata_path.clear();
  }
}
