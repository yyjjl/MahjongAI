cmake_minimum_required(VERSION 2.8)
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

project(MahJong)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pg")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

set(MAHJONG_LIB_DIR "${CMAKE_SOURCE_DIR}/third-party/ChineseOfficialMahjongHelper/Classes/mahjong-algorithm/")

include_directories(
  "${CMAKE_SOURCE_DIR}/include"
  "${CMAKE_SOURCE_DIR}/mahjong-game"
  "${MAHJONG_LIB_DIR}")

SET(MAHJONG_GB_LIB_FILES
  "${MAHJONG_LIB_DIR}/fan_calculator.cpp"
  "${MAHJONG_LIB_DIR}/shanten.cpp")

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third-party/pybind11)
include_directories(${PYTHON_INCLUDE_DIRS})

add_library(MahjongGame STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/mahjong-game/message.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/mahjong-game/player.cc
  ${MAHJONG_GB_LIB_FILES})
set_target_properties(MahjongGame PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

file(GLOB INTERFACE_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/mahjong-game/PyMahjong*.cc)
pybind11_add_module(PyMahjong MODULE ${INTERFACE_SOURCE_FILES})
target_link_libraries(PyMahjong PRIVATE MahjongGame)

add_custom_command(
  TARGET PyMahjong
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:PyMahjong> ${CMAKE_CURRENT_SOURCE_DIR}/mahjong-python/$<TARGET_FILE_NAME:PyMahjong>
)

