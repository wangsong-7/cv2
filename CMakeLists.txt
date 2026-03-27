cmake_minimum_required(VERSION 3.15)
project(image_preprocess)

set(CMAKE_CXX_STANDARD 17)
add_compile_options("$<$<CXX_COMPILER_ID:GNU>:-finput-charset=UTF-8>")

# 找到 OpenCV
find_package(OpenCV REQUIRED)

add_executable(task1 "Measurement Task/task1.cpp")
add_executable(task2 "Measurement Task/task2.cpp")
add_executable(corner_test "day2/corner_test.cpp")
add_executable(image_preprocess "day1/image_preprocess.cpp")
target_include_directories(image_preprocess PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(image_preprocess PRIVATE ${OpenCV_LIBS})
# ─────────────────────────────────────────────
# 把 OpenCV 链接给需要它的目
target_link_libraries(task1 PRIVATE ${OpenCV_LIBS})
target_link_libraries(task2 PRIVATE ${OpenCV_LIBS})
target_link_libraries(corner_test PRIVATE ${OpenCV_LIBS})