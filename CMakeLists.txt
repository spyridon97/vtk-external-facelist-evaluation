cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(vtk-external-facelist-evaluation LANGUAGES C CXX)

if (DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  # Enable CUDA language
  enable_language(CUDA)
  message(STATUS "CUDA language enabled.")
endif ()
if (DEFINED CMAKE_HIP_ARCHITECTURES AND NOT CMAKE_HIP_ARCHITECTURES STREQUAL "")
  # Enable HIP language
  enable_language(HIP)
  message(STATUS "HIP language enabled.")
endif ()

# VTK flags
set(VTK_SMP_IMPLEMENTATION_TYPE TBB CACHE STRING "" FORCE)
#set(VTK_USE_KOKKOS ON CACHE BOOL "" FORCE)
#set(VTK_KOKKOS_BACKEND HIP CACHE STRING "" FORCE)
set(VTK_MODULE_ENABLE_VTK_vtkvtkm YES CACHE BOOL "" FORCE)
set(VTK_MODULE_ENABLE_VTK_AcceleratorsVTKmCore YES CACHE BOOL "" FORCE)
set(VTK_MODULE_ENABLE_VTK_AcceleratorsVTKmDataModel YES CACHE BOOL "" FORCE)
set(VTK_MODULE_ENABLE_VTK_AcceleratorsVTKmFilters YES CACHE BOOL "" FORCE)

set(FETCHCONTENT_QUIET OFF)
include(FetchContent)

FetchContent_Declare(
  VTK
  GIT_REPOSITORY https://gitlab.kitware.com/vtk/vtk.git
  GIT_TAG v9.4.0
  GIT_PROGRESS ON
  SOURCE_DIR VTK
)
FetchContent_MakeAvailable(VTK)

FetchContent_Declare(
  CLI11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
  GIT_TAG v2.4.2
  GIT_PROGRESS ON
  SOURCE_DIR CLI11
)
FetchContent_MakeAvailable(CLI11)

set(headers
  src/CellFaceMinMaxPointId.h

  src/ExternalFacesHashCountFnv1a.h
  src/ExternalFacesHashCountMinPointId.h
  src/ExternalFacesHashFightFnv1a.h
  src/ExternalFacesHashFightMinPointId.h
  src/ExternalFacesHashSortFnv1a.h
  src/ExternalFacesHashSortMinPointId.h

  src/vtkGeometryFilterSClassifier.h
  src/vtkDataSetSurfaceFilterSHash.h
  src/vtkGeometryFilterPClassifier.h
  src/vtkDataSetSurfaceFilterSHash.h

  src/YamlWriter.h
  src/Arguments.h
)

set(sources
  src/vtkGeometryFilterSClassifier.cxx
  src/vtkDataSetSurfaceFilterSHash.cxx
  src/vtkGeometryFilterPClassifier.cxx
  src/vtkGeometryFilterPHash.cxx

  src/Arguments.cxx
)

set(device_sources
  src/ExternalFacelistEvaluation.cxx)

add_executable(${PROJECT_NAME} ${headers} ${sources} ${device_sources})

target_include_directories(${PROJECT_NAME} PRIVATE ${VTKm_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PUBLIC
  VTK::CommonCore VTK::IOXML VTK::FiltersGeometry VTK::vtkvtkm VTK::AcceleratorsVTKmDataModel CLI11::CLI11)
vtkm_add_target_information(${PROJECT_NAME}
  DROP_UNUSED_SYMBOLS
  MODIFY_CUDA_FLAGS
  DEVICE_SOURCES ${device_sources}
)

set_source_files_properties(${headers}
  PROPERTIES HEADER_FILE_ONLY TRUE
)
