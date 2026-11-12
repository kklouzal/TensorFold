cmake_minimum_required(VERSION 3.27)
project(OwnedC10dSocketRegression LANGUAGES CXX)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "The owned socket regression requires the selected Linux SDK")
endif()
if(NOT EXISTS "${PYTORCH_SOURCE_DIR}/test/cpp/c10d/SocketTest.cpp")
  message(FATAL_ERROR "Set PYTORCH_SOURCE_DIR to the exact selected owned framework checkout")
endif()
find_package(Torch CONFIG REQUIRED)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
add_subdirectory("${PYTORCH_SOURCE_DIR}/third_party/googletest" "${CMAKE_CURRENT_BINARY_DIR}/googletest" EXCLUDE_FROM_ALL)

# Compile the actual owned test and production source, never editable mirrors.
add_executable(OwnedSocketTest
  "${PYTORCH_SOURCE_DIR}/test/cpp/c10d/SocketTest.cpp"
  "${PYTORCH_SOURCE_DIR}/torch/csrc/distributed/c10d/socket.cpp")
target_compile_features(OwnedSocketTest PRIVATE cxx_std_20)
# Match the actual selected torch_cpu/socket.cpp fmt interface from its build graph.
# Torch does not export a separate linked fmt library for this owned source unit.
target_compile_definitions(OwnedSocketTest PRIVATE FMT_HEADER_ONLY=1 "FMT_NO_UNIQUE_ADDRESS=[[no_unique_address]]")
target_include_directories(OwnedSocketTest BEFORE PRIVATE "${PYTORCH_SOURCE_DIR}/third_party/fmt/include")
target_include_directories(OwnedSocketTest PRIVATE ${TORCH_INCLUDE_DIRS} "${PYTORCH_SOURCE_DIR}")
separate_arguments(OWNED_TORCH_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(OwnedSocketTest PRIVATE ${OWNED_TORCH_FLAGS})
target_link_options(OwnedSocketTest PRIVATE "-Wl,--wrap=getnameinfo")
target_link_libraries(OwnedSocketTest PRIVATE ${TORCH_LIBRARIES} gtest_main)
enable_testing()
add_test(NAME OwnedSocketTest COMMAND OwnedSocketTest)
