CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
PROJECT(KittiFlow)

ADD_DEFINITIONS(-std=c++11)
SET(CMAKE_BUILD_TYPE "Debug")

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -fPIC ") #-fsingle-precision-constant
else(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3 -O0 -fPIC -rdynamic -export-dynamic")
endif(CMAKE_BUILD_TYPE STREQUAL "Release")

SET(SRC_LIST
src/main.cc
utils/SignalBase.cc
)
SET(INCLUDE_LIST
    ${PROJECT_SOURCE_DIR}/inc/
    ${PROJECT_SOURCE_DIR}/dataset/
    ${PROJECT_SOURCE_DIR}/utils/
)

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)


find_package(OpenCV REQUIRED)
find_package (glog REQUIRED)
find_package(PNG REQUIRED)

MESSAGE(STATUS "\nThis is PROJECT name: " ${PROJECT_NAME})
MESSAGE(STATUS "This is PROJECT_SOURCE_DIR: " ${PROJECT_SOURCE_DIR})
MESSAGE(STATUS "This is BINARY dir: " ${PROJECT_BINARY_DIR})
MESSAGE(STATUS "This is exe dir: " ${EXECUTABLE_OUTPUT_PATH})
MESSAGE(STATUS "This is CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS})
MESSAGE(STATUS "This is CMAKE_BUILD_TYPE: " ${CMAKE_BUILD_TYPE})
MESSAGE(STATUS "This is SRC_LIST: " ${SRC_LIST})
MESSAGE(STATUS "This is INCLUDE_LIST: " ${INCLUDE_LIST})
MESSAGE(STATUS "This is opencv inlcude: " ${OpenCV_INCLUDE_DIRS})
MESSAGE(STATUS "This is opencv lib: " ${OpenCV_LIBS})

ADD_EXECUTABLE(KittiFlow ${SRC_LIST})

INCLUDE_DIRECTORIES(
    ${INCLUDE_LIST}
    ${OpenCV_INCLUDE_DIRS}
    ${PNG_INCLUDE_DIR}
    /usr/local/include
    )

link_directories(
    /usr/local/lib
    ${LIBRARY_OUTPUT_PATH}
)

TARGET_LINK_LIBRARIES(KittiFlow
-lglog
-lgflags
-ldl
# -fopenmp
${OpenCV_LIBS}
${PNG_LIBRARY}
)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(${PROJECT_NAME}
    # OpenMP::OpenMP_CXX is an imported target with INTERFACE_**
    OpenMP::OpenMP_CXX )
endif()