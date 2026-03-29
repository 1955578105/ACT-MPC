#include "xbox.h"

#include <csignal>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>

namespace
{
volatile std::sig_atomic_t g_keep_running = 1;

void HandleSignal(int)
{
  g_keep_running = 0;
}

void ResetMap(xbox_map_t &state)
{
  state.time = 0;
  state.a = 0;
  state.b = 0;
  state.x = 0;
  state.y = 0;
  state.lb = 0;
  state.rb = 0;
  state.start = 0;
  state.back = 0;
  state.home = 0;
  state.lo = 0;
  state.ro = 0;
  state.lx = 0;
  state.ly = 0;
  state.rx = 0;
  state.ry = 0;
  state.lt = 0;
  state.rt = 0;
  state.xx = 0;
  state.yy = 0;
}

const char *ButtonName(unsigned char number)
{
  switch (number)
  {
  case XBOX_BUTTON_A:
    return "A";
  case XBOX_BUTTON_B:
    return "B";
  case XBOX_BUTTON_X:
    return "X";
  case XBOX_BUTTON_Y:
    return "Y";
  case XBOX_BUTTON_LB:
    return "LB";
  case XBOX_BUTTON_RB:
    return "RB";
  case XBOX_BUTTON_START:
    return "START";
  case XBOX_BUTTON_BACK:
    return "BACK";
  case XBOX_BUTTON_HOME:
    return "HOME";
  case XBOX_BUTTON_LO:
    return "L3";
  case XBOX_BUTTON_RO:
    return "R3";
  default:
    return "UNKNOWN_BUTTON";
  }
}

const char *AxisName(unsigned char number)
{
  switch (number)
  {
  case XBOX_AXIS_LX:
    return "LX";
  case XBOX_AXIS_LY:
    return "LY";
  case XBOX_AXIS_RX:
    return "RX";
  case XBOX_AXIS_RY:
    return "RY";
  case XBOX_AXIS_LT:
    return "LT";
  case XBOX_AXIS_RT:
    return "RT";
  case XBOX_AXIS_XX:
    return "DPAD_X";
  case XBOX_AXIS_YY:
    return "DPAD_Y";
  default:
    return "UNKNOWN_AXIS";
  }
}

bool UpdateButtonState(xbox_map_t &state, unsigned char number, int value)
{
  switch (number)
  {
  case XBOX_BUTTON_A:
    state.a = value;
    return true;
  case XBOX_BUTTON_B:
    state.b = value;
    return true;
  case XBOX_BUTTON_X:
    state.x = value;
    return true;
  case XBOX_BUTTON_Y:
    state.y = value;
    return true;
  case XBOX_BUTTON_LB:
    state.lb = value;
    return true;
  case XBOX_BUTTON_RB:
    state.rb = value;
    return true;
  case XBOX_BUTTON_START:
    state.start = value;
    return true;
  case XBOX_BUTTON_BACK:
    state.back = value;
    return true;
  case XBOX_BUTTON_HOME:
    state.home = value;
    return true;
  case XBOX_BUTTON_LO:
    state.lo = value;
    return true;
  case XBOX_BUTTON_RO:
    state.ro = value;
    return true;
  default:
    return false;
  }
}

bool UpdateAxisState(xbox_map_t &state, unsigned char number, int value)
{
  switch (number)
  {
  case XBOX_AXIS_LX:
    state.lx = value;
    return true;
  case XBOX_AXIS_LY:
    state.ly = value;
    return true;
  case XBOX_AXIS_RX:
    state.rx = value;
    return true;
  case XBOX_AXIS_RY:
    state.ry = value;
    return true;
  case XBOX_AXIS_LT:
    state.lt = value;
    return true;
  case XBOX_AXIS_RT:
    state.rt = value;
    return true;
  case XBOX_AXIS_XX:
    state.xx = value;
    return true;
  case XBOX_AXIS_YY:
    state.yy = value;
    return true;
  default:
    return false;
  }
}

void PrintUsage(const char *program)
{
  std::cout << "Usage: " << program << " [/dev/input/jsX]\n";
  std::cout << "Default device: /dev/input/js0\n";
}

const char *EventTypeName(unsigned char event_type)
{
  switch (event_type)
  {
  case JS_EVENT_BUTTON:
    return "BUTTON";
  case JS_EVENT_AXIS:
    return "AXIS";
  default:
    return "OTHER";
  }
}

void PrintEvent(const xbox_map_t &state, const js_event &event, bool is_init_event)
{
  const unsigned char event_type = event.type & ~JS_EVENT_INIT;

  std::cout << "\r\033[2K";
  std::cout << (is_init_event ? "[INIT] " : "[EVT ] ");
  std::cout << "time=" << std::setw(8) << event.time
            << " raw_type=" << std::setw(6) << EventTypeName(event_type)
            << " raw_num=" << std::setw(2) << static_cast<int>(event.number)
            << " raw_val=" << std::setw(6) << event.value
            << " map=";

  if (event_type == JS_EVENT_BUTTON)
  {
    std::cout << std::setw(6) << ButtonName(event.number);
  }
  else if (event_type == JS_EVENT_AXIS)
  {
    std::cout << std::setw(7) << AxisName(event.number);
  }
  else
  {
    std::cout << "UNKNOWN";
  }

  std::cout << " | BTN"
            << " A:" << state.a.load()
            << " B:" << state.b.load()
            << " X:" << state.x.load()
            << " Y:" << state.y.load()
            << " LB:" << state.lb.load()
            << " RB:" << state.rb.load()
            << " START:" << state.start.load()
            << " BACK:" << state.back.load()
            << " HOME:" << state.home.load()
            << " L3:" << state.lo.load()
            << " R3:" << state.ro.load();

  std::cout << " | AXS"
            << " LX:" << std::setw(6) << state.lx.load()
            << " LY:" << std::setw(6) << state.ly.load()
            << " RX:" << std::setw(6) << state.rx.load()
            << " RY:" << std::setw(6) << state.ry.load()
            << " LT:" << std::setw(6) << state.lt.load()
            << " RT:" << std::setw(6) << state.rt.load()
            << " DX:" << std::setw(6) << state.xx.load()
            << " DY:" << std::setw(6) << state.yy.load()
            << std::flush;
}
} // namespace

int main(int argc, char **argv)
{
  const char *device = "/dev/input/js0";
  if (argc > 2)
  {
    PrintUsage(argv[0]);
    return 1;
  }
  if (argc == 2)
  {
    if (std::strcmp(argv[1], "-h") == 0 || std::strcmp(argv[1], "--help") == 0)
    {
      PrintUsage(argv[0]);
      return 0;
    }
    device = argv[1];
  }

  std::signal(SIGINT, HandleSignal);
  std::signal(SIGTERM, HandleSignal);

  xbox_map_t state;
  ResetMap(state);

  const int xbox_fd = xbox_open(device);
  if (xbox_fd < 0)
  {
    std::cerr << "Failed to open joystick device: " << device << "\n";
    return 1;
  }

  std::cout << "Listening on " << device << "\n";
  std::cout << "Press Ctrl+C to exit.\n";
  std::cout << "Tip: press one button at a time and watch raw_num first.\n";

  while (g_keep_running)
  {
    js_event event;
    const int len = read(xbox_fd, &event, sizeof(event));
    if (len < 0)
    {
      if (errno == EINTR && g_keep_running)
      {
        continue;
      }

      std::perror("read");
      xbox_close(xbox_fd);
      return 1;
    }

    if (len != static_cast<int>(sizeof(event)))
    {
      std::cerr << "Short read from joystick device.\n";
      xbox_close(xbox_fd);
      return 1;
    }

    state.time = event.time;

    const bool is_init_event = (event.type & JS_EVENT_INIT) != 0;
    const unsigned char event_type = event.type & ~JS_EVENT_INIT;

    if (event_type == JS_EVENT_BUTTON)
    {
      UpdateButtonState(state, event.number, event.value);
      PrintEvent(state, event, is_init_event);
      continue;
    }

    if (event_type == JS_EVENT_AXIS)
    {
      UpdateAxisState(state, event.number, event.value);
      PrintEvent(state, event, is_init_event);
      continue;
    }

    PrintEvent(state, event, is_init_event);
  }

  std::cout << "\nExit joystick test.\n";
  xbox_close(xbox_fd);
  return 0;
}
