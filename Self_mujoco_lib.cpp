#include "Self_mujoco_lib.h"
#include <algorithm>
#include <cstdio>
#include <GLFW/glfw3.h>

/*
  mujoco自定义函数库
*/
namespace mujo
{
  namespace
  {
    bool EnsureGlfwInitialized()
    {
      static bool initialized = false;
      static bool success = false;
      if (!initialized)
      {
        initialized = true;
        success = (glfwInit() == GLFW_TRUE);
        if (!success)
        {
          std::cout << "failed to initialize GLFW for offscreen renderer" << std::endl;
        }
      }
      return success;
    }

    void FlipRgbBuffer(std::vector<unsigned char> &rgb, int width, int height)
    {
      const int row_bytes = 3 * width;
      for (int r = 0; r < height / 2; ++r)
      {
        unsigned char *top_row = rgb.data() + row_bytes * r;
        unsigned char *bottom_row = rgb.data() + row_bytes * (height - 1 - r);
        std::swap_ranges(top_row, top_row + row_bytes, bottom_row);
      }
    }

    void FlipDepthBuffer(std::vector<float> &depth, int width, int height)
    {
      const int row_width = width;
      for (int r = 0; r < height / 2; ++r)
      {
        float *top_row = depth.data() + row_width * r;
        float *bottom_row = depth.data() + row_width * (height - 1 - r);
        std::swap_ranges(top_row, top_row + row_width, bottom_row);
      }
    }
  } // namespace

  OffscreenCameraRenderer::OffscreenCameraRenderer()
  {
    mjv_defaultScene(&scene_);
    mjv_defaultOption(&option_);
    option_.flags[mjVIS_RANGEFINDER] = 0;
    mjr_defaultContext(&context_);
  }

  OffscreenCameraRenderer::~OffscreenCameraRenderer()
  {
    ResetRenderResources();
    if (window_)
    {
      glfwDestroyWindow(window_);
      window_ = nullptr;
    }
  }

  void OffscreenCameraRenderer::ResetRenderResources()
  {
    if (initialized_)
    {
      mjv_freeScene(&scene_);
      mjr_freeContext(&context_);
      mjv_defaultScene(&scene_);
      mjr_defaultContext(&context_);
      initialized_ = false;
    }
    model_ = nullptr;
    width_ = 0;
    height_ = 0;
  }

  bool OffscreenCameraRenderer::Initialize(const mjModel *model, int width, int height, int max_geom)
  {
    if (!model)
    {
      std::cout << "offscreen renderer initialize failed: model is null" << std::endl;
      return false;
    }
    if (width <= 0 || height <= 0)
    {
      std::cout << "offscreen renderer initialize failed: invalid image size" << std::endl;
      return false;
    }
    if (!EnsureGlfwInitialized())
    {
      return false;
    }

    if (!window_)
    {
      glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
      glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_FALSE);
      window_ = glfwCreateWindow(width, height, "mujoco_offscreen", nullptr, nullptr);
      if (!window_)
      {
        std::cout << "failed to create hidden GLFW window for offscreen renderer" << std::endl;
        return false;
      }
    }

    glfwMakeContextCurrent(window_);

    const bool need_rebuild = !initialized_ || model_ != model || max_geom_ != max_geom;
    if (need_rebuild)
    {
      ResetRenderResources();
      mjv_defaultOption(&option_);
      option_.flags[mjVIS_RANGEFINDER] = 0;
      mjv_makeScene(model, &scene_, max_geom);
      mjr_makeContext(model, &context_, mjFONTSCALE_150);
      max_geom_ = max_geom;
    }

    mjr_setBuffer(mjFB_OFFSCREEN, &context_);
    if (context_.currentBuffer != mjFB_OFFSCREEN)
    {
      std::cout << "failed to switch to MuJoCo offscreen framebuffer" << std::endl;
      ResetRenderResources();
      return false;
    }

    if (!initialized_ || width_ != width || height_ != height)
    {
      mjr_resizeOffscreen(width, height, &context_);
      mjr_setBuffer(mjFB_OFFSCREEN, &context_);
    }

    initialized_ = true;
    model_ = model;
    width_ = width;
    height_ = height;
    return true;
  }

  bool OffscreenCameraRenderer::Render(const mjModel *model, mjData *data, const std::string &camera_name,
                                       CameraFrame &frame, bool read_depth, bool flip_vertical)
  {
    if (!Initialize(model, frame.width, frame.height, max_geom_))
    {
      return false;
    }
    if (!data)
    {
      std::cout << "offscreen renderer failed: data is null" << std::endl;
      return false;
    }

    const int camera_id = mj_name2id(model, mjOBJ_CAMERA, camera_name.c_str());
    if (camera_id < 0)
    {
      std::cout << "camera not found: " << camera_name << std::endl;
      return false;
    }

    glfwMakeContextCurrent(window_);

    mjvCamera camera;
    mjv_defaultCamera(&camera);
    camera.type = mjCAMERA_FIXED;
    camera.fixedcamid = camera_id;

    mjr_setBuffer(mjFB_OFFSCREEN, &context_);

    mjv_updateScene(model, data, &option_, nullptr, &camera, mjCAT_ALL, &scene_);

    mjrRect viewport = {0, 0, frame.width, frame.height};
    mjr_render(viewport, &scene_, &context_);

    frame.rgb.resize(3 * frame.width * frame.height);
    float *depth_ptr = nullptr;
    if (read_depth)
    {
      frame.depth.resize(frame.width * frame.height);
      depth_ptr = frame.depth.data();
    }
    else
    {
      frame.depth.clear();
    }

    mjr_readPixels(frame.rgb.data(), depth_ptr, viewport, &context_);

    if (flip_vertical)
    {
      FlipRgbBuffer(frame.rgb, frame.width, frame.height);
      if (read_depth)
      {
        FlipDepthBuffer(frame.depth, frame.width, frame.height);
      }
    }

    return true;
  }

  // 根据 传感器的名 返回 传感器数据
  std::vector<float> get_sensor_data(const mjModel *model, mjData *data,
                                     const std::string &sensor_name)
  {
    // 1. 先获取 id
    int sensor_id = mj_name2id(model, mjOBJ_SENSOR, sensor_name.c_str());
    if (sensor_id == -1)
    {
      std::cout << "no found sensor" << std::endl;
      return std::vector<float>();
    }
    // 2.根据id 获取 起始地址
    int data_pos = model->sensor_adr[sensor_id];
    // 3. 根据id  获取数据大小（维度）
    std::vector<float> sensor_data(model->sensor_dim[sensor_id]);
    for (int i = 0; i < sensor_data.size(); i++)
    {
      sensor_data[i] = data->sensordata[data_pos + i];
    }
    return sensor_data;
  }

  std::vector<float> get_lidar_scan(const mjModel *model, mjData *data,
                                    const std::string &sensor_prefix, int beam_count)
  {
    std::vector<float> scan;
    if (beam_count <= 0)
    {
      return scan;
    }

    scan.resize(beam_count, -1.0f);
    for (int i = 0; i < beam_count; ++i)
    {
      char sensor_name[128];
      std::snprintf(sensor_name, sizeof(sensor_name), "%s_%03d", sensor_prefix.c_str(), i);
      std::vector<float> beam = get_sensor_data(model, data, sensor_name);
      if (!beam.empty())
      {
        scan[i] = beam[0];
      }
    }

    return scan;
  }

  std::vector<float> get_lidar_scan_90(const mjModel *model, mjData *data)
  {
    return get_lidar_scan(model, data, "lidar", 90);
  }

  bool render_camera_rgb(const mjModel *model, mjData *data, const std::string &camera_name,
                         int width, int height, std::vector<unsigned char> &rgb,
                         bool flip_vertical)
  {
    CameraFrame frame;
    frame.width = width;
    frame.height = height;

    thread_local OffscreenCameraRenderer renderer;
    if (!renderer.Render(model, data, camera_name, frame, false, flip_vertical))
    {
      return false;
    }

    rgb = std::move(frame.rgb);
    return true;
  }

  bool render_camera_frame(const mjModel *model, mjData *data, const std::string &camera_name,
                           int width, int height, CameraFrame &frame,
                           bool read_depth, bool flip_vertical)
  {
    frame.width = width;
    frame.height = height;

    thread_local OffscreenCameraRenderer renderer;
    return renderer.Render(model, data, camera_name, frame, read_depth, flip_vertical);
  }

};
