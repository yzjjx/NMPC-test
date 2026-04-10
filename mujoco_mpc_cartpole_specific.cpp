#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
#include "mpc_controller.h"
}

namespace {

// ======================= 用户可改参数 =======================
constexpr double kControlDt = 0.04;  // 与 MATLAB 中 dt 保持一致
constexpr double kInitX     = 0.00;
constexpr double kInitXd    = 0.05;
constexpr double kInitPhi   = 0.03;
constexpr double kInitPhid  = 0.01;
// ===========================================================

mjModel* g_model = nullptr;
mjData*  g_data  = nullptr;
GLFWwindow* g_window = nullptr;

mjvCamera g_cam;
mjvOption g_opt;
mjvScene  g_scene;
mjrContext g_con;

bool g_button_left   = false;
bool g_button_middle = false;
bool g_button_right  = false;
double g_lastx = 0.0;
double g_lasty = 0.0;
bool g_paused = false;

int g_slider_jid  = -1;
int g_hinge_jid   = -1;
int g_slider_qadr = -1;
int g_hinge_qadr  = -1;
int g_slider_vadr = -1;
int g_hinge_vadr  = -1;
int g_act_id      = -1;

// 用于显示/日志
std::array<double, 4> g_last_state {0.0, 0.0, 0.0, 0.0};
double g_last_u = 0.0;
double g_next_ctrl_time = 0.0;

struct CasadiMpcWrapper {
  CasadiMpcWrapper() {
    M_incref();
    arg_.resize(M_SZ_ARG, nullptr);
    res_.resize(M_SZ_RES, nullptr);
    iw_.resize(M_SZ_IW, 0);
    w_.resize(M_SZ_W, 0.0);

    // M 只有一个输入 p 和一个输出 u_opt
    arg_[0] = p_.data();
    res_[0] = u_.data();
  }

  ~CasadiMpcWrapper() {
    M_decref();
  }

  double Solve(const std::array<double, 4>& x) {
    p_ = x;

    int mem = M_checkout();
    if (mem < 0) {
      throw std::runtime_error("CasADi M_checkout() 失败");
    }

    int flag = M(arg_.data(), res_.data(), iw_.data(), w_.data(), mem);
    M_release(mem);

    if (flag != 0) {
      throw std::runtime_error("CasADi M(...) 求值失败");
    }

    return u_[0];
  }

 private:
  std::array<casadi_real, 4> p_ {0.0, 0.0, 0.0, 0.0};
  std::array<casadi_real, 1> u_ {0.0};
  std::vector<const casadi_real*> arg_;
  std::vector<casadi_real*> res_;
  std::vector<casadi_int> iw_;
  std::vector<casadi_real> w_;
};

CasadiMpcWrapper* g_mpc = nullptr;

void ApplyInitialState() {
  mj_resetData(g_model, g_data);

  g_data->qpos[g_slider_qadr] = kInitX;
  g_data->qpos[g_hinge_qadr]  = kInitPhi;
  g_data->qvel[g_slider_vadr] = kInitXd;
  g_data->qvel[g_hinge_vadr]  = kInitPhid;

  g_last_state = {kInitX, kInitXd, kInitPhi, kInitPhid};
  g_last_u = 0.0;
  g_next_ctrl_time = 0.0;

  mj_forward(g_model, g_data);
}

std::array<double, 4> ReadStateFromMujoco(const mjModel* m, const mjData* d) {
  (void)m;
  std::array<double, 4> x;
  x[0] = d->qpos[g_slider_qadr];
  x[1] = d->qvel[g_slider_vadr];
  x[2] = d->qpos[g_hinge_qadr];
  x[3] = d->qvel[g_hinge_vadr];
  return x;
}

void ComputeAndSetControl(const mjModel* m, mjData* d) {
  try {
    std::array<double, 4> x = ReadStateFromMujoco(m, d);
    double u = g_mpc->Solve(x);

    // 你的 XML 中 actuator: gear='50' 且 ctrlrange='-1 1'
    // MATLAB 模型中 actual_force = 50*u
    // 因此这里直接把 u 写到 ctrl 即可，不需要再乘 50。
    u = std::clamp(u, -1.0, 1.0);
    d->ctrl[g_act_id] = u;

    g_last_state = x;
    g_last_u = u;
  } catch (const std::exception& e) {
    std::cerr << "控制器求值失败: " << e.what() << std::endl;
    d->ctrl[g_act_id] = 0.0;
    g_last_u = 0.0;
  }
}

void ControlCallback(const mjModel* m, mjData* d) {
  // 按 0.04 s 更新一次控制律，其余仿真子步保持上一次控制量
  if (d->time + 1e-12 >= g_next_ctrl_time) {
    ComputeAndSetControl(m, d);
    while (d->time + 1e-12 >= g_next_ctrl_time) {
      g_next_ctrl_time += kControlDt;
    }
  } else {
    d->ctrl[g_act_id] = g_last_u;
  }
}

void Keyboard(GLFWwindow* window, int key, int scancode, int act, int mods) {
  (void)window; (void)scancode; (void)mods;
  if (act == GLFW_PRESS) {
    if (key == GLFW_KEY_BACKSPACE) {
      ApplyInitialState();
    } else if (key == GLFW_KEY_SPACE) {
      g_paused = !g_paused;
    } else if (key == GLFW_KEY_ESCAPE) {
      glfwSetWindowShouldClose(g_window, GLFW_TRUE);
    }
  }
}

void MouseButton(GLFWwindow* window, int button, int act, int mods) {
  (void)mods;
  g_button_left   = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
  g_button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS);
  g_button_right  = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);
  glfwGetCursorPos(window, &g_lastx, &g_lasty);
}

void MouseMove(GLFWwindow* window, double xpos, double ypos) {
  if (!g_button_left && !g_button_middle && !g_button_right) {
    return;
  }

  double dx = xpos - g_lastx;
  double dy = ypos - g_lasty;
  g_lastx = xpos;
  g_lasty = ypos;

  int width = 0, height = 0;
  glfwGetWindowSize(window, &width, &height);
  if (height == 0) return;

  bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                    glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);

  mjtMouse action;
  if (g_button_right) {
    action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
  } else if (g_button_left) {
    action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
  } else {
    action = mjMOUSE_ZOOM;
  }

  mjv_moveCamera(g_model, action, dx / height, dy / height, &g_scene, &g_cam);
}

void Scroll(GLFWwindow* window, double xoffset, double yoffset) {
  (void)window; (void)xoffset;
  mjv_moveCamera(g_model, mjMOUSE_ZOOM, 0.0, -0.05 * yoffset, &g_scene, &g_cam);
}

void RenderOverlay(mjrRect viewport) {
  std::ostringstream left, right;
  left << "Space\nBackspace\nEsc\nTime\nCtrl dt\nControl u\nCart x\nCart x_dot\nPole phi\nPole phi_dot";

  right.setf(std::ios::fixed);
  right.precision(4);
  right << (g_paused ? "Resume" : "Pause") << "\n"
        << "Reset\n"
        << "Quit\n"
        << g_data->time << " s\n"
        << kControlDt << " s\n"
        << g_last_u << "\n"
        << g_last_state[0] << "\n"
        << g_last_state[1] << "\n"
        << g_last_state[2] << "\n"
        << g_last_state[3];

  const std::string l = left.str();
  const std::string r = right.str();
  mjr_overlay(mjFONT_NORMAL, mjGRID_TOPLEFT, viewport, l.c_str(), r.c_str(), &g_con);
}

void InitModelIdsOrThrow(const mjModel* m) {
  g_slider_jid = mj_name2id(m, mjOBJ_JOINT, "slider");
  g_hinge_jid  = mj_name2id(m, mjOBJ_JOINT, "hinge");
  g_act_id     = mj_name2id(m, mjOBJ_ACTUATOR, "slide");

  if (g_slider_jid < 0 || g_hinge_jid < 0 || g_act_id < 0) {
    throw std::runtime_error("没有在 XML 中找到 slider / hinge / slide，请检查名字是否一致");
  }

  g_slider_qadr = m->jnt_qposadr[g_slider_jid];
  g_hinge_qadr  = m->jnt_qposadr[g_hinge_jid];
  g_slider_vadr = m->jnt_dofadr[g_slider_jid];
  g_hinge_vadr  = m->jnt_dofadr[g_hinge_jid];
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "用法: " << argv[0] << " your_cartpole.xml" << std::endl;
    return 1;
  }

  const char* xml_path = argv[1];

  char error[1000] = "";
  g_model = mj_loadXML(xml_path, nullptr, error, sizeof(error));
  if (!g_model) {
    std::cerr << "MuJoCo XML 加载失败:\n" << error << std::endl;
    return 1;
  }

  g_data = mj_makeData(g_model);
  if (!g_data) {
    std::cerr << "mj_makeData 失败" << std::endl;
    mj_deleteModel(g_model);
    return 1;
  }

  try {
    InitModelIdsOrThrow(g_model);
    CasadiMpcWrapper mpc;
    g_mpc = &mpc;

    ApplyInitialState();

    // 安装控制回调：MuJoCo 会在需要控制输入时调用它
    mjcb_control = ControlCallback;

    if (!glfwInit()) {
      throw std::runtime_error("glfwInit() 失败");
    }

    g_window = glfwCreateWindow(1400, 900, "MuJoCo + CasADi MPC Cartpole", nullptr, nullptr);
    if (!g_window) {
      throw std::runtime_error("glfwCreateWindow() 失败");
    }

    glfwMakeContextCurrent(g_window);
    glfwSwapInterval(1);

    glfwSetKeyCallback(g_window, Keyboard);
    glfwSetCursorPosCallback(g_window, MouseMove);
    glfwSetMouseButtonCallback(g_window, MouseButton);
    glfwSetScrollCallback(g_window, Scroll);

    mjv_defaultCamera(&g_cam);
    mjv_defaultOption(&g_opt);
    mjv_defaultScene(&g_scene);
    mjr_defaultContext(&g_con);

    mjv_makeScene(g_model, &g_scene, 2000);
    mjr_makeContext(g_model, &g_con, mjFONTSCALE_150);

    int fixed_cam_id = mj_name2id(g_model, mjOBJ_CAMERA, "fixed");
    if (fixed_cam_id >= 0) {
      g_cam.type = mjCAMERA_FIXED;
      g_cam.fixedcamid = fixed_cam_id;
    } else {
      g_cam.type = mjCAMERA_FREE;
      g_cam.azimuth = 90.0;
      g_cam.elevation = -20.0;
      g_cam.distance = 3.0;
      g_cam.lookat[0] = 0.0;
      g_cam.lookat[1] = 0.0;
      g_cam.lookat[2] = 0.0;
    }

    while (!glfwWindowShouldClose(g_window)) {
      // 以接近实时的方式推进仿真
      if (!g_paused) {
        const mjtNum simstart = g_data->time;
        while (g_data->time - simstart < 1.0 / 60.0) {
          mj_step(g_model, g_data);
        }
      }

      int width = 0, height = 0;
      glfwGetFramebufferSize(g_window, &width, &height);
      mjrRect viewport = {0, 0, width, height};

      mjv_updateScene(g_model, g_data, &g_opt, nullptr, &g_cam, mjCAT_ALL, &g_scene);
      mjr_render(viewport, &g_scene, &g_con);
      RenderOverlay(viewport);

      glfwSwapBuffers(g_window);
      glfwPollEvents();
    }

    mjcb_control = nullptr;
    mjr_freeContext(&g_con);
    mjv_freeScene(&g_scene);
    glfwDestroyWindow(g_window);
    glfwTerminate();
  } catch (const std::exception& e) {
    std::cerr << "程序异常: " << e.what() << std::endl;
    mjcb_control = nullptr;
    if (g_window) {
      glfwDestroyWindow(g_window);
      glfwTerminate();
    }
    if (g_data) mj_deleteData(g_data);
    if (g_model) mj_deleteModel(g_model);
    return 1;
  }

  mj_deleteData(g_data);
  mj_deleteModel(g_model);
  return 0;
}
