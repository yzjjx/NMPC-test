cmake_minimum_required(VERSION 3.16)
project(NMPC_cartpole)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MuJoCo 路径
set(MUJOCO_DIR "/home/yzj_robot/Cpp_pac/mujoco-3.6.0")

# 头文件
include_directories(
    ${MUJOCO_DIR}/include
    /usr/include
)

# 你的可执行文件
add_executable(nmpc_cartpole
    nmpc_cartpole_robust.cpp
    mpc_controller.c
)

# 手动查找库
find_library(MUJOCO_LIB mujoco PATHS ${MUJOCO_DIR}/lib ${MUJOCO_DIR}/build/lib ${MUJOCO_DIR}/bin NO_DEFAULT_PATH)
find_library(GLFW_LIB glfw PATHS /usr/lib /usr/lib/x86_64-linux-gnu)

# 输出检查信息
message(STATUS "MUJOCO_LIB = ${MUJOCO_LIB}")
message(STATUS "GLFW_LIB   = ${GLFW_LIB}")

# 链接
target_link_libraries(nmpc_cartpole
    ${MUJOCO_LIB}
    ${GLFW_LIB}
    dl
    pthread
    m
)