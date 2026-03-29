// Copyright 2021 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <array>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <new>
#include <random>
#include <string>
#include <thread>
#include <unistd.h>
#include "qpOASES.hpp"
#include <mujoco/mujoco.h>
#include "glfw_adapter.h"
#include "simulate.h"
#include "array_safety.h"
#include "Quad.h"
#include "platform_ui_adapter.h"
#include "Self_mujoco_lib.h"
#define MUJOCO_PLUGIN_DIR "mujoco_plugin"

extern "C"
{
#if defined(_WIN32) || defined(__CYGWIN__)
#include <windows.h>
#else
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif
#include <sys/errno.h>
#include <unistd.h>
#endif
}
using namespace qpOASES;

void myController(const mjModel *m, mjData *d)
{
  Quad::SystemControl::Control_Step(m, d, MPC_T);
}
namespace
{
  namespace mj = ::mujoco;
  namespace mju = ::mujoco::sample_util;

  // constants
  const double syncMisalign = 0.1;       // maximum mis-alignment before re-sync (simulation seconds)
  const double simRefreshFraction = 0.7; // fraction of refresh available for simulation
  const int kErrorLength = 1024;         // load error string length
  enum class RuntimeMode
  {
    kCapture,
    kDeploy,
  };

  constexpr RuntimeMode kRuntimeMode = RuntimeMode::kDeploy;
  constexpr int kCommandMode = (kRuntimeMode == RuntimeMode::kDeploy) ? 1 : 0;
  constexpr float kEpisodeCaptureHz = 20.0f;
  constexpr int kEpisodeCaptureSeconds = 15;
  const std::string kEpisodeCaptureRoot = "/home/Actdog/capture";

  struct EpisodeCapturePaths
  {
    std::string dir;
    std::string file;
  };

  EpisodeCapturePaths MakeNextEpisodeCapturePaths()
  {
    namespace fs = std::filesystem;

    if (kRuntimeMode != RuntimeMode::kDeploy)
    {
      return {"", ""};
    }

    std::error_code ec;
    fs::create_directories(kEpisodeCaptureRoot, ec);

    for (int index = 0; index < 10000; ++index)
    {
      std::ostringstream name;
      name << "episode_" << std::setw(4) << std::setfill('0') << index;
      fs::path episode_dir = fs::path(kEpisodeCaptureRoot) / name.str();
      if (!fs::exists(episode_dir))
      {
        return {episode_dir.string(), name.str() + ".json"};
      }
    }

    return {fs::path(kEpisodeCaptureRoot) / "episode_overflow",
            "episode_overflow.json"};
  }

  const EpisodeCapturePaths kEpisodeCapturePaths = MakeNextEpisodeCapturePaths();

  // model and data
  mjModel *m = nullptr;
  mjData *d = nullptr;

  struct MovingTargetState
  {
    int body_id = -2;
    int mocap_id = -2;
    double last_update_time = 0.0;
    double next_velocity_change_time = 0.0;
    double velocity_x = 0.0;
    double velocity_y = 0.0;
    double pos_x = 0.0;
    double pos_y = 0.0;
    double pos_z = 0.30;
    bool initialized = false;
  };

  struct ObstacleState
  {
    std::array<int, 96> geom_ids = {
        -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2,
        -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2,
        -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2,
        -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2,
        -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2,
        -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2,
        -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2,
        -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2};
    bool initialized = false;
  };

  struct SharedSpawnState
  {
    double robot_x = 6.0;
    double robot_y = 0.0;
    double target_x = 6.7;
    double target_y = 0.0;
    bool initialized = false;
  };

  std::mt19937 &SceneRng()
  {
    static std::mt19937 rng(std::random_device{}());
    return rng;
  }

  MovingTargetState moving_target_state;
  ObstacleState obstacle_state;
  SharedSpawnState shared_spawn_state;

  void ResetSceneRandomizationState()
  {
    moving_target_state = MovingTargetState{};
    obstacle_state = ObstacleState{};
    shared_spawn_state = SharedSpawnState{};
  }

  void EnsureSharedSpawnPoint()
  {
    if (shared_spawn_state.initialized)
    {
      return;
    }

    std::uniform_real_distribution<double> x_dist(1.0, 11.0);
    std::uniform_real_distribution<double> y_dist(-4.5, 4.5);
    std::uniform_real_distribution<double> forward_dist(0.55, 0.95);
    std::uniform_real_distribution<double> lateral_dist(-0.25, 0.25);

    shared_spawn_state.robot_x = x_dist(SceneRng());
    shared_spawn_state.robot_y = y_dist(SceneRng());
    shared_spawn_state.target_x = std::clamp(
        shared_spawn_state.robot_x + forward_dist(SceneRng()), 0.45, 11.55);
    shared_spawn_state.target_y = std::clamp(
        shared_spawn_state.robot_y + lateral_dist(SceneRng()), -5.0, 5.0);
    shared_spawn_state.initialized = true;
  }

  void ApplySharedSpawnToRobot(mjData *data)
  {
    if (!data)
    {
      return;
    }

    EnsureSharedSpawnPoint();
    data->qpos[0] = shared_spawn_state.robot_x;
    data->qpos[1] = shared_spawn_state.robot_y;
    data->qpos[2] = 0.27;
  }

  void SampleTargetVelocity(MovingTargetState &state, double current_time)
  {
    std::uniform_real_distribution<double> angle_dist(0.0, 2.0 * M_PI);
    std::uniform_real_distribution<double> speed_dist(0.08, 0.16);
    std::uniform_real_distribution<double> duration_dist(2.0, 4.5);

    const double angle = angle_dist(SceneRng());
    const double speed = speed_dist(SceneRng());
    state.velocity_x = speed * std::cos(angle);
    state.velocity_y = speed * std::sin(angle);
    state.next_velocity_change_time = current_time + duration_dist(SceneRng());
  }

  void InitializeObstaclePositions(mjModel *model)
  {
    if (!model)
    {
      return;
    }

    EnsureSharedSpawnPoint();

    if (!obstacle_state.initialized)
    {
      for (int i = 0; i < static_cast<int>(obstacle_state.geom_ids.size()); ++i)
      {
        char name[32];
        std::snprintf(name, sizeof(name), "obstacle_cyl_%d", i);
        obstacle_state.geom_ids[i] = mj_name2id(model, mjOBJ_GEOM, name);
      }
      obstacle_state.initialized = true;
    }

    std::uniform_real_distribution<double> x_dist(0.9, 11.2);
    std::uniform_real_distribution<double> y_dist(-4.8, 4.8);
    std::uniform_real_distribution<double> radius_dist(0.12, 0.2);
    std::uniform_real_distribution<double> half_height_dist(0.24, 0.4);

    std::vector<std::array<double, 2>> used_positions;
    used_positions.push_back({shared_spawn_state.robot_x, shared_spawn_state.robot_y});
    used_positions.push_back({shared_spawn_state.target_x, shared_spawn_state.target_y});

    for (int geom_id : obstacle_state.geom_ids)
    {
      if (geom_id < 0)
      {
        continue;
      }

      const double radius = radius_dist(SceneRng());
      const double half_height = half_height_dist(SceneRng());
      std::array<double, 2> candidate = {1.5, 0.0};
      bool accepted = false;

      for (int attempt = 0; attempt < 500 && !accepted; ++attempt)
      {
        candidate = {x_dist(SceneRng()), y_dist(SceneRng())};
        accepted = true;

        for (const auto &used : used_positions)
        {
          const double dx = candidate[0] - used[0];
          const double dy = candidate[1] - used[1];
          const double min_dist = (&used == &used_positions.front()) ? 1.35 : 0.55 + 2.6 * radius;
          if (dx * dx + dy * dy < min_dist * min_dist)
          {
            accepted = false;
            break;
          }
        }
      }

      model->geom_size[3 * geom_id + 0] = radius;
      model->geom_size[3 * geom_id + 1] = half_height;
      model->geom_pos[3 * geom_id + 0] = candidate[0];
      model->geom_pos[3 * geom_id + 1] = candidate[1];
      model->geom_pos[3 * geom_id + 2] = half_height;
      used_positions.push_back(candidate);
    }
  }

  void UpdateMovingTarget(const mjModel *model, mjData *data)
  {
    if (!model || !data)
    {
      return;
    }

    if (moving_target_state.body_id == -2)
    {
      moving_target_state.body_id = mj_name2id(model, mjOBJ_BODY, "moving_target");
      if (moving_target_state.body_id >= 0)
      {
        moving_target_state.mocap_id = model->body_mocapid[moving_target_state.body_id];
      }
      else
      {
        moving_target_state.mocap_id = -1;
      }
    }

    if (moving_target_state.body_id < 0 || moving_target_state.mocap_id < 0)
    {
      return;
    }

    const double t = data->time;
    if (!moving_target_state.initialized)
    {
      EnsureSharedSpawnPoint();
      moving_target_state.initialized = true;
      moving_target_state.last_update_time = t;
      moving_target_state.pos_x = shared_spawn_state.target_x;
      moving_target_state.pos_y = shared_spawn_state.target_y;
      moving_target_state.pos_z = 0.30;
      SampleTargetVelocity(moving_target_state, t);
    }

    const double dt = std::max(0.0, t - moving_target_state.last_update_time);
    moving_target_state.last_update_time = t;

    if (t >= moving_target_state.next_velocity_change_time)
    {
      SampleTargetVelocity(moving_target_state, t);
    }

    moving_target_state.pos_x += moving_target_state.velocity_x * dt;
    moving_target_state.pos_y += moving_target_state.velocity_y * dt;

    const double xmin = 0.35;
    const double xmax = 11.75;
    const double ymin = -5.15;
    const double ymax = 5.15;

    bool bounced = false;
    if (moving_target_state.pos_x < xmin)
    {
      moving_target_state.pos_x = xmin;
      bounced = true;
    }
    else if (moving_target_state.pos_x > xmax)
    {
      moving_target_state.pos_x = xmax;
      bounced = true;
    }

    if (moving_target_state.pos_y < ymin)
    {
      moving_target_state.pos_y = ymin;
      bounced = true;
    }
    else if (moving_target_state.pos_y > ymax)
    {
      moving_target_state.pos_y = ymax;
      bounced = true;
    }

    if (bounced)
    {
      SampleTargetVelocity(moving_target_state, t);
    }

    data->mocap_pos[3 * moving_target_state.mocap_id + 0] = moving_target_state.pos_x;
    data->mocap_pos[3 * moving_target_state.mocap_id + 1] = moving_target_state.pos_y;
    data->mocap_pos[3 * moving_target_state.mocap_id + 2] = moving_target_state.pos_z;
    data->mocap_quat[4 * moving_target_state.mocap_id + 0] = 1.0;
    data->mocap_quat[4 * moving_target_state.mocap_id + 1] = 0.0;
    data->mocap_quat[4 * moving_target_state.mocap_id + 2] = 0.0;
    data->mocap_quat[4 * moving_target_state.mocap_id + 3] = 0.0;
  }

  using Seconds = std::chrono::duration<double>;

  //---------------------------------------- plugin handling -----------------------------------------

  // return the path to the directory containing the current executable
  // used to determine the location of auto-loaded plugin libraries
  std::string getExecutableDir()
  {
#if defined(_WIN32) || defined(__CYGWIN__)
    constexpr char kPathSep = '\\';
    std::string realpath = [&]() -> std::string
    {
      std::unique_ptr<char[]> realpath(nullptr);
      DWORD buf_size = 128;
      bool success = false;
      while (!success)
      {
        realpath.reset(new (std::nothrow) char[buf_size]);
        if (!realpath)
        {
          std::cerr << "cannot allocate memory to store executable path\n";
          return "";
        }

        DWORD written = GetModuleFileNameA(nullptr, realpath.get(), buf_size);
        if (written < buf_size)
        {
          success = true;
        }
        else if (written == buf_size)
        {
          // realpath is too small, grow and retry
          buf_size *= 2;
        }
        else
        {
          std::cerr << "failed to retrieve executable path: " << GetLastError() << "\n";
          return "";
        }
      }
      return realpath.get();
    }();
#else
    constexpr char kPathSep = '/';
#if defined(__APPLE__)
    std::unique_ptr<char[]> buf(nullptr);
    {
      std::uint32_t buf_size = 0;
      _NSGetExecutablePath(nullptr, &buf_size);
      buf.reset(new char[buf_size]);
      if (!buf)
      {
        std::cerr << "cannot allocate memory to store executable path\n";
        return "";
      }
      if (_NSGetExecutablePath(buf.get(), &buf_size))
      {
        std::cerr << "unexpected error from _NSGetExecutablePath\n";
      }
    }
    const char *path = buf.get();
#else
    const char *path = "/proc/self/exe";
#endif
    std::string realpath = [&]() -> std::string
    {
      std::unique_ptr<char[]> realpath(nullptr);
      std::uint32_t buf_size = 128;
      bool success = false;
      while (!success)
      {
        realpath.reset(new (std::nothrow) char[buf_size]);
        if (!realpath)
        {
          std::cerr << "cannot allocate memory to store executable path\n";
          return "";
        }

        std::size_t written = readlink(path, realpath.get(), buf_size);
        if (written < buf_size)
        {
          realpath.get()[written] = '\0';
          success = true;
        }
        else if (written == -1)
        {
          if (errno == EINVAL)
          {
            // path is already not a symlink, just use it
            return path;
          }

          std::cerr << "error while resolving executable path: " << strerror(errno) << '\n';
          return "";
        }
        else
        {
          // realpath is too small, grow and retry
          buf_size *= 2;
        }
      }
      return realpath.get();
    }();
#endif

    if (realpath.empty())
    {
      return "";
    }

    for (std::size_t i = realpath.size() - 1; i > 0; --i)
    {
      if (realpath.c_str()[i] == kPathSep)
      {
        return realpath.substr(0, i);
      }
    }

    // don't scan through the entire file system's root
    return "";
  }

  // scan for libraries in the plugin directory to load additional plugins
  void scanPluginLibraries()
  {
    // check and print plugins that are linked directly into the executable
    int nplugin = mjp_pluginCount();
    if (nplugin)
    {
      std::printf("Built-in plugins:\n");
      for (int i = 0; i < nplugin; ++i)
      {
        std::printf("    %s\n", mjp_getPluginAtSlot(i)->name);
      }
    }

    // define platform-specific strings
#if defined(_WIN32) || defined(__CYGWIN__)
    const std::string sep = "\\";
#else
    const std::string sep = "/";
#endif

    // try to open the ${EXECDIR}/MUJOCO_PLUGIN_DIR directory
    // ${EXECDIR} is the directory containing the simulate binary itself
    // MUJOCO_PLUGIN_DIR is the MUJOCO_PLUGIN_DIR preprocessor macro
    const std::string executable_dir = getExecutableDir();
    if (executable_dir.empty())
    {
      return;
    }

    const std::string plugin_dir = getExecutableDir() + sep + MUJOCO_PLUGIN_DIR;
    mj_loadAllPluginLibraries(
        plugin_dir.c_str(), +[](const char *filename, int first, int count)
                            {
        std::printf("Plugins registered by library '%s':\n", filename);
        for (int i = first; i < first + count; ++i) {
          std::printf("    %s\n", mjp_getPluginAtSlot(i)->name);
        } });
  }

  //------------------------------------------- simulation -------------------------------------------

  const char *Diverged(int disableflags, const mjData *d)
  {
    if (disableflags & mjDSBL_AUTORESET)
    {
      for (mjtWarning w : {mjWARN_BADQACC, mjWARN_BADQVEL, mjWARN_BADQPOS})
      {
        if (d->warning[w].number > 0)
        {
          return mju_warningText(w, d->warning[w].lastinfo);
        }
      }
    }
    return nullptr;
  }

  mjModel *LoadModel(const char *file, mj::Simulate &sim)
  {
    // this copy is needed so that the mju::strlen call below compiles
    char filename[mj::Simulate::kMaxFilenameLength];
    mju::strcpy_arr(filename, file);

    // make sure filename is not empty
    if (!filename[0])
    {
      return nullptr;
    }

    // load and compile
    char loadError[kErrorLength] = "";
    mjModel *mnew = 0;
    auto load_start = mj::Simulate::Clock::now();
    if (mju::strlen_arr(filename) > 4 &&
        !std::strncmp(filename + mju::strlen_arr(filename) - 4, ".mjb",
                      mju::sizeof_arr(filename) - mju::strlen_arr(filename) + 4))
    {
      mnew = mj_loadModel(filename, nullptr);
      if (!mnew)
      {
        mju::strcpy_arr(loadError, "could not load binary model");
      }
    }
    else
    {
      mnew = mj_loadXML(filename, nullptr, loadError, kErrorLength);

      // remove trailing newline character from loadError
      if (loadError[0])
      {
        int error_length = mju::strlen_arr(loadError);
        if (loadError[error_length - 1] == '\n')
        {
          loadError[error_length - 1] = '\0';
        }
      }
    }
    auto load_interval = mj::Simulate::Clock::now() - load_start;
    double load_seconds = Seconds(load_interval).count();

    if (!mnew)
    {
      std::printf("%s\n", loadError);
      mju::strcpy_arr(sim.load_error, loadError);
      return nullptr;
    }

    // compiler warning: print and pause
    if (loadError[0])
    {
      // mj_forward() below will print the warning message
      std::printf("Model compiled, but simulation warning (paused):\n  %s\n", loadError);
      sim.run = 0;
    }

    // if no error and load took more than 1/4 seconds, report load time
    else if (load_seconds > 0.25)
    {
      mju::sprintf_arr(loadError, "Model loaded in %.2g seconds", load_seconds);
    }

    mju::strcpy_arr(sim.load_error, loadError);

    return mnew;
  }

  // simulate in background thread (while rendering in main thread)
  void PhysicsLoop(mj::Simulate &sim)
  {
    // cpu-sim syncronization point
    std::chrono::time_point<mj::Simulate::Clock> syncCPU;
    mjtNum syncSim = 0;

    // run until asked to exit
    while (!sim.exitrequest.load())
    {
      if (sim.droploadrequest.load())
      {
        sim.LoadMessage(sim.dropfilename);
        mjModel *mnew = LoadModel(sim.dropfilename, sim);
        sim.droploadrequest.store(false);

        mjData *dnew = nullptr;
        if (mnew)
          dnew = mj_makeData(mnew);
        if (dnew)
        {
          sim.Load(mnew, dnew, sim.dropfilename);

          // lock the sim mutex
          const std::unique_lock<std::recursive_mutex> lock(sim.mtx);

          mj_deleteData(d);
          mj_deleteModel(m);

          m = mnew;
          d = dnew;
          mj_forward(m, d);
        }
        else
        {
          sim.LoadMessageClear();
        }
      }

      if (sim.uiloadrequest.load())
      {
        sim.uiloadrequest.fetch_sub(1);
        sim.LoadMessage(sim.filename);
        mjModel *mnew = LoadModel(sim.filename, sim);
        mjData *dnew = nullptr;
        if (mnew)
          dnew = mj_makeData(mnew);
        if (dnew)
        {
          sim.Load(mnew, dnew, sim.filename);

          // lock the sim mutex
          const std::unique_lock<std::recursive_mutex> lock(sim.mtx);

          mj_deleteData(d);
          mj_deleteModel(m);

          m = mnew;
          d = dnew;
          mj_forward(m, d);
        }
        else
        {
          sim.LoadMessageClear();
        }
      }

      // sleep for 1 ms or yield, to let main thread run
      //  yield results in busy wait - which has better timing but kills battery life
      if (sim.run && sim.busywait)
      {
        std::this_thread::yield();
      }
      else
      {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }

      {
        // lock the sim mutex
        const std::unique_lock<std::recursive_mutex> lock(sim.mtx);

        // run only if model is present
        if (m)
        {
          // running
          if (sim.run)
          {
            bool stepped = false;

            // record cpu time at start of iteration
            const auto startCPU = mj::Simulate::Clock::now();

            // elapsed CPU and simulation time since last sync
            const auto elapsedCPU = startCPU - syncCPU;
            double elapsedSim = d->time - syncSim;

            // requested slow-down factor
            double slowdown = 100 / sim.percentRealTime[sim.real_time_index];

            // misalignment condition: distance from target sim time is bigger than syncmisalign
            bool misaligned =
                std::abs(Seconds(elapsedCPU).count() / slowdown - elapsedSim) > syncMisalign;

            // out-of-sync (for any reason): reset sync times, step
            if (elapsedSim < 0 || elapsedCPU.count() < 0 || syncCPU.time_since_epoch().count() == 0 ||
                misaligned || sim.speed_changed)
            {
              // re-sync
              syncCPU = startCPU;
              syncSim = d->time;
              sim.speed_changed = false;

              // run single step, let next iteration deal with timing
              UpdateMovingTarget(m, d);
              mj_step(m, d);

              const char *message = Diverged(m->opt.disableflags, d);
              if (message)
              {
                sim.run = 0;
                mju::strcpy_arr(sim.load_error, message);
              }
              else
              {
                stepped = true;
              }
            }

            // in-sync: step until ahead of cpu
            else
            {
              bool measured = false;
              mjtNum prevSim = d->time;

              double refreshTime = simRefreshFraction / sim.refresh_rate;

              // step while sim lags behind cpu and within refreshTime
              while (Seconds((d->time - syncSim) * slowdown) < mj::Simulate::Clock::now() - syncCPU &&
                     mj::Simulate::Clock::now() - startCPU < Seconds(refreshTime))
              {
                // measure slowdown before first step
                if (!measured && elapsedSim)
                {
                  sim.measured_slowdown =
                      std::chrono::duration<double>(elapsedCPU).count() / elapsedSim;
                  measured = true;
                }

                // inject noise
                sim.InjectNoise();
                UpdateMovingTarget(m, d);
                // call mj_step
                mj_step(m, d);

                const char *message = Diverged(m->opt.disableflags, d);
                if (message)
                {
                  sim.run = 0;
                  mju::strcpy_arr(sim.load_error, message);
                }
                else
                {
                  stepped = true;
                }

                // break if reset
                if (d->time < prevSim)
                {
                  break;
                }
              }
            }

            // save current state to history buffer
            if (stepped)
            {
              sim.AddToHistory();
              if (kRuntimeMode == RuntimeMode::kCapture)
              {
                const bool episode_done =
                    EpisodeCaptureData(kEpisodeCaptureHz, kEpisodeCaptureSeconds,
                                       m, d, kEpisodeCapturePaths.dir, kEpisodeCapturePaths.file);
                if (episode_done)
                {
                  sim.run = 0;
                  sim.exitrequest.store(true);
                }
              }
            }
          }

          // paused
          else
          {
            // run mj_forward, to update rendering and joint sliders
            UpdateMovingTarget(m, d);
            mj_forward(m, d);
            sim.speed_changed = true;
          }
        }
      } // release std::lock_guard<std::mutex>
    }
  }
} // namespace

//-------------------------------------- physics_thread --------------------------------------------

void PhysicsThread(mj::Simulate *sim, const char *filename)
{

  // request loadmodel if file given (otherwise drag-and-drop)
  if (filename != nullptr)
  {
    sim->LoadMessage(filename);
    m = LoadModel(filename, *sim);
    if (m)
    {
      // lock the sim mutex
      const std::unique_lock<std::recursive_mutex> lock(sim->mtx);

      d = mj_makeData(m);
      ResetSceneRandomizationState();
      InitializeObstaclePositions(m);
      Quad::KeyboardIns::ReceiveCommandMode = kCommandMode;
      Quad::SystemControl::System_Init(m, d, MPC_T);
      mj_resetDataKeyframe(m, d, 0);
      ApplySharedSpawnToRobot(d);
      UpdateMovingTarget(m, d);

      mjcb_control = myController;
    }
    if (d)
    {

      sim->Load(m, d, filename);

      // lock the sim mutex
      // std::thread thread(&threadtask, sim, m, d);
      // thread.join();
      const std::unique_lock<std::recursive_mutex> lock(sim->mtx);

      // std::thread mpcthread(&threadMpc, sim);
      // mpcthread.detach();
      if (m->ncam > 0)
      {
        sim->cam.type = mjCAMERA_FIXED;
        sim->cam.fixedcamid = 0;
        sim->camera = 2;
      }
      UpdateMovingTarget(m, d);
      mj_forward(m, d);
    }
    else
    {
      sim->LoadMessageClear();
    }
  }

  PhysicsLoop(*sim);

  // delete everything we allocated
  mj_deleteData(d);
  mj_deleteModel(m);
}

//------------------------------------------ main --------------------------------------------------

// machinery for replacing command line error by a macOS dialog box when running under Rosetta
#if defined(__APPLE__) && defined(__AVX__)
extern void DisplayErrorDialogBox(const char *title, const char *msg);
static const char *rosetta_error_msg = nullptr;
__attribute__((used, visibility("default"))) extern "C" void _mj_rosettaError(const char *msg)
{
  rosetta_error_msg = msg;
}
#endif

// run event loop
int main(int argc, char **argv)
{

  // display an error if running on macOS under Rosetta 2
#if defined(__APPLE__) && defined(__AVX__)
  if (rosetta_error_msg)
  {
    DisplayErrorDialogBox("Rosetta 2 is not supported", rosetta_error_msg);
    std::exit(1);
  }
#endif

  // print version, check compatibility
  std::printf("MuJoCo version %s\n", mj_versionString());
  if (mjVERSION_HEADER != mj_version())
  {
    mju_error("Headers and library have different versions");
  }

  // scan for libraries in the plugin directory to load additional plugins
  scanPluginLibraries();

  mjvCamera cam;
  mjv_defaultCamera(&cam);

  mjvOption opt;
  mjv_defaultOption(&opt);
  opt.flags[mjVIS_RANGEFINDER] = 0;

  mjvPerturb pert;
  mjv_defaultPerturb(&pert);

  // simulate object encapsulates the UI
  auto sim = std::make_unique<mj::Simulate>(
      std::make_unique<mj::GlfwAdapter>(),
      &cam, &opt, &pert, /* is_passive = */ false);
  sim->ui0_enable = 0;
  sim->ui1_enable = 0;
  sim->fullscreen = 0;
  sim->camera = 2;
  sim->cam.type = mjCAMERA_FIXED;
  sim->cam.fixedcamid = 0;

  const char *filename = nullptr;
  filename = "../unitree_mujoco/unitree_robots/go2/scene_terrain.xml";
  // if (argc >  1) {
  //   filename = argv[1];
  // }

  // std::thread mpcthread(threadMpc, sim.get());
  //  start physics thread

  Quad::KeyboardIns::ReceiveCommandMode = kCommandMode;
  std::thread physicsthreadhandle(&PhysicsThread, sim.get(), filename);
  std::thread Keythread;
  if (kCommandMode == 0)
  {
    Keythread = std::thread(Quad::KeyboardIns::Update_ins);
  }

  // start simulation UI loop (blocking call)

  sim->RenderLoop();

  physicsthreadhandle.join();
  if (Keythread.joinable())
  {
    Keythread.detach();
  }

  // mpcthread.join();

  return 0;
}
