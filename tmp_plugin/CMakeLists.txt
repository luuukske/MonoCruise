cmake_minimum_required(VERSION 3.20)

project(monocruise_tmp VERSION 1.0.1 LANGUAGES CXX RC)

if (NOT MSVC)
    message(FATAL_ERROR "The TruckersMP client is Windows x64 only; build with MSVC.")
endif()

add_library(monocruise_tmp SHARED
    src/main.cpp
    src/version.rc
)

target_compile_features(monocruise_tmp PRIVATE cxx_std_17)
target_include_directories(monocruise_tmp PRIVATE vendor/truckersmp_sdk/include)
target_compile_definitions(monocruise_tmp PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX UNICODE _UNICODE)

# Static CRT: nothing to install beside the DLL. /Brepro keeps the bytes, and so the SHA, reproducible.
set_target_properties(monocruise_tmp PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded")
target_compile_options(monocruise_tmp PRIVATE /W4 /permissive- /utf-8 /Brepro)
target_link_options(monocruise_tmp PRIVATE /Brepro)
