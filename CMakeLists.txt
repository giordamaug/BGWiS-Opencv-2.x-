cmake_minimum_required(VERSION 2.8)

project( cdwisard )

# Find OpenMP
find_package(OpenMP)

if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	list( APPEND ThirdParty_LIBS ${OpenCV_LIBS} )
    include_directories( ${OpenCV_INCLUDE_DIRS} )
endif()

# Find OpenMP
find_package(OpenCV REQUIRED)
message( STATUS "OpenCV LIBS ${OpenCV_LIBS}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

# Build CDWis Library
# Build CDWisard program
set(BGWIZSRC_NAME CDwizlib/bgwis.cpp)
set(MAINSRC_NAME cdwisard.cpp)

set(CDWIZLIB_NAME cdwiz_$ENV{USER})
set(CDWIZEXE_NAME cdwisard_$ENV{USER})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories( /opt/local/include/opencv2 )

include_directories(./CDwizlib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ./CDwizlib)
add_library(${CDWIZLIB_NAME} SHARED ${BGWIZSRC_NAME} CDwizlib/wisard.cpp CDwizlib/wnet_lib.cpp)
target_link_libraries( ${CDWIZLIB_NAME} ${OpenCV_LIBS})

link_directories(./CDwizlib)
add_executable(${CDWIZEXE_NAME} ${MAINSRC_NAME})
target_link_libraries(${CDWIZEXE_NAME} ${OpenCV_LIBS} ${CDWIZLIB_NAME})
