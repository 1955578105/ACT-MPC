#ifndef __MUJO_H
#define __MUJO_H

#include <iostream>
#include <string>
#include <vector>
#include <mujoco/mjmodel.h>
#include <mujoco/mujoco.h>

struct GLFWwindow;

namespace mujo
{

  struct CameraFrame
  {
    int width = 0;
    int height = 0;
    std::vector<unsigned char> rgb;
    std::vector<float> depth;
  };

  class OffscreenCameraRenderer
  {
  public:
    OffscreenCameraRenderer();
    ~OffscreenCameraRenderer();

    OffscreenCameraRenderer(const OffscreenCameraRenderer &) = delete;
    OffscreenCameraRenderer &operator=(const OffscreenCameraRenderer &) = delete;

    bool Initialize(const mjModel *model, int width, int height, int max_geom = 2000);
    bool Render(const mjModel *model, mjData *data, const std::string &camera_name,
                CameraFrame &frame, bool read_depth = false, bool flip_vertical = true);

  private:
    void ResetRenderResources();

    GLFWwindow *window_ = nullptr;
    mjvScene scene_;
    mjvOption option_;
    mjrContext context_;
    const mjModel *model_ = nullptr;
    int width_ = 0;
    int height_ = 0;
    int max_geom_ = 2000;
    bool initialized_ = false;
  };

  std::vector<float> get_sensor_data(const mjModel *model, mjData *data,
                                     const std::string &sensor_name);

  std::vector<float> get_lidar_scan(const mjModel *model, mjData *data,
                                    const std::string &sensor_prefix, int beam_count);

  std::vector<float> get_lidar_scan_90(const mjModel *model, mjData *data);

  bool render_camera_rgb(const mjModel *model, mjData *data, const std::string &camera_name,
                         int width, int height, std::vector<unsigned char> &rgb,
                         bool flip_vertical = true);

  bool render_camera_frame(const mjModel *model, mjData *data, const std::string &camera_name,
                           int width, int height, CameraFrame &frame,
                           bool read_depth = false, bool flip_vertical = true);
};

#endif
