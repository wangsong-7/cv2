cmake_minimum_required(VERSION 3.15)
project(CV2)

set(CMAKE_CXX_STANDARD 17)

# 找到 OpenCV
find_package(OpenCV REQUIRED)

add_executable(task1 "Measurement Task/task1.cpp")
add_executable(task2 "Measurement Task/task2.cpp")
# add_executable(test1 "day1/test1.cpp")

# 把 OpenCV 链接给需要它的目标
target_link_libraries(task1 PRIVATE ${OpenCV_LIBS})
target_link_libraries(task2 PRIVATE ${OpenCV_LIBS})