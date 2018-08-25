cmake_minimum_required(VERSION 3.0)
project(sharedLibsDemo)                 # create a project with the given name

add_executable(soa soa.cpp)             # compile "main.cpp" the the executable called "main"
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT soa)

include (GenerateExportHeader)