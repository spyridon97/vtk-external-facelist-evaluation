//============================================================================
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//============================================================================
#ifndef vtk_m_exec_CellFaceMinMaxPointId_h
#define vtk_m_exec_CellFaceMinMaxPointId_h

#include "vtkm/exec/CellFace.h"

namespace vtkm
{
namespace exec
{

/// \brief Returns the min point id of a cell face
/// Given information about a cell face and the global point indices for that cell, returns a
/// vtkm::Id that contains the minimum point id for that face.
template <typename CellShapeTag, typename GlobalPointIndicesVecType>
static inline VTKM_EXEC vtkm::ErrorCode CellFaceMinPointId(vtkm::IdComponent faceIndex,
  CellShapeTag shape, const GlobalPointIndicesVecType& globalPointIndicesVec,
  vtkm::Id& minFacePointId)
{
  vtkm::IdComponent numPointsInFace;
  minFacePointId = { -1 };
  VTKM_RETURN_ON_ERROR(vtkm::exec::CellFaceNumberOfPoints(faceIndex, shape, numPointsInFace));
  if (numPointsInFace == 0)
  {
    // An invalid face. We should already have gotten an error from
    // CellFaceNumberOfPoints.
    return vtkm::ErrorCode::InvalidFaceId;
  }

  detail::CellFaceTables table;
  minFacePointId = globalPointIndicesVec[table.PointsInFace(shape.Id, faceIndex, 0)];
  vtkm::Id nextPoint = globalPointIndicesVec[table.PointsInFace(shape.Id, faceIndex, 1)];
  if (nextPoint < minFacePointId)
  {
    minFacePointId = nextPoint;
  }
  nextPoint = globalPointIndicesVec[table.PointsInFace(shape.Id, faceIndex, 2)];
  if (nextPoint < minFacePointId)
  {
    minFacePointId = nextPoint;
  }

  // Check the rest of the points to see if they are in the lowest 3
  for (vtkm::IdComponent pointIndex = 3; pointIndex < numPointsInFace; pointIndex++)
  {
    nextPoint = globalPointIndicesVec[table.PointsInFace(shape.Id, faceIndex, pointIndex)];
    if (nextPoint < minFacePointId)
    {
      minFacePointId = nextPoint;
    }
  }

  return vtkm::ErrorCode::Success;
}
/// \brief Returns the max point id of a cell face
/// Given information about a cell face and the global point indices for that cell, returns a
/// vtkm::Id that contains the maximum point id for that face.
template <typename CellShapeTag, typename GlobalPointIndicesVecType>
static inline VTKM_EXEC vtkm::ErrorCode CellFaceMaxPointId(vtkm::IdComponent faceIndex,
  CellShapeTag shape, const GlobalPointIndicesVecType& globalPointIndicesVec,
  vtkm::Id& maxFacePointId)
{
  vtkm::IdComponent numPointsInFace;
  maxFacePointId = { -1 };
  VTKM_RETURN_ON_ERROR(vtkm::exec::CellFaceNumberOfPoints(faceIndex, shape, numPointsInFace));
  if (numPointsInFace == 0)
  {
    // An invalid face. We should already have gotten an error from
    // CellFaceNumberOfPoints.
    return vtkm::ErrorCode::InvalidFaceId;
  }

  detail::CellFaceTables table;
  maxFacePointId = globalPointIndicesVec[table.PointsInFace(shape.Id, faceIndex, 0)];
  vtkm::Id nextPoint = globalPointIndicesVec[table.PointsInFace(shape.Id, faceIndex, 1)];
  if (nextPoint > maxFacePointId)
  {
    maxFacePointId = nextPoint;
  }
  nextPoint = globalPointIndicesVec[table.PointsInFace(shape.Id, faceIndex, 2)];
  if (nextPoint > maxFacePointId)
  {
    maxFacePointId = nextPoint;
  }

  // Check the rest of the points to see if they are in the lowest 3
  for (vtkm::IdComponent pointIndex = 3; pointIndex < numPointsInFace; pointIndex++)
  {
    nextPoint = globalPointIndicesVec[table.PointsInFace(shape.Id, faceIndex, pointIndex)];
    if (nextPoint > maxFacePointId)
    {
      maxFacePointId = nextPoint;
    }
  }

  return vtkm::ErrorCode::Success;
}
}
} // namespace vtkm::exec

#endif // vtk_m_exec_CellFaceMinMaxPointId_h
