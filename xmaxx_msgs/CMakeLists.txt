cmake_minimum_required(VERSION 3.8)
project(xmaxx_msgs)

set(CMAKE_CXX_STANDARD 17)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(std_msgs REQUIRED)

#add_compile_options(-Wall -Wextra -Wpedantic -pedantic-errors -Wcast-align)
#add_compile_options(-Wconversion -Wsign-conversion -Wold-style-cast)

set(msg_files
  msg/XmaxxTelem.msg
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  DEPENDENCIES
  builtin_interfaces
  std_msgs
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
