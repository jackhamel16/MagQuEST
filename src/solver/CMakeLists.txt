#include("${CMAKE_CURRENT_LIST_DIR}/RHS/CMakeLists.txt")

target_sources(quest
  PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/solver.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/solver.h"
    "${CMAKE_CURRENT_LIST_DIR}/jfnk.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/jfnk.h"
    "${CMAKE_CURRENT_LIST_DIR}/gmres.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/gmres.h"
)
