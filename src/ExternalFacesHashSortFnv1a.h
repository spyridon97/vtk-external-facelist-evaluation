//============================================================================
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//============================================================================
#ifndef viskores_worklet_ExternalFacesHashSortFnv1a_h
#define viskores_worklet_ExternalFacesHashSortFnv1a_h

#include <viskores/CellShape.h>
#include <viskores/Hash.h>
#include <viskores/Math.h>

#include <viskores/exec/CellFace.h>

#include <viskores/cont/Algorithm.h>
#include <viskores/cont/ArrayCopy.h>
#include <viskores/cont/ArrayCopyDevice.h>
#include <viskores/cont/ArrayGetValues.h>
#include <viskores/cont/ArrayHandle.h>
#include <viskores/cont/ArrayHandleConcatenate.h>
#include <viskores/cont/ArrayHandleConstant.h>
#include <viskores/cont/ArrayHandleGroupVec.h>
#include <viskores/cont/ArrayHandleGroupVecVariable.h>
#include <viskores/cont/ArrayHandleIndex.h>
#include <viskores/cont/ArrayHandlePermutation.h>
#include <viskores/cont/ArrayHandleTransform.h>
#include <viskores/cont/ArrayHandleView.h>
#include <viskores/cont/CellSetExplicit.h>
#include <viskores/cont/ConvertNumComponentsToOffsets.h>
#include <viskores/cont/DataSet.h>
#include <viskores/cont/Field.h>
#include <viskores/cont/Timer.h>

#include <viskores/worklet/DispatcherMapTopology.h>
#include <viskores/worklet/DispatcherReduceByKey.h>
#include <viskores/worklet/Keys.h>
#include <viskores/worklet/ScatterCounting.h>
#include <viskores/worklet/WorkletMapTopology.h>
#include <viskores/worklet/WorkletReduceByKey.h>

#include "YamlWriter.h"

namespace viskores
{
namespace worklet
{

struct ExternalFacesHashSortFnv1a
{
  // Worklet that returns the number of faces for each cell/shape
  class NumFacesPerCell : public viskores::worklet::WorkletVisitCellsWithPoints
  {
  public:
    using ControlSignature = void(CellSetIn inCellSet, FieldOut numFacesInCell);
    using ExecutionSignature = void(CellShape, _2);
    using InputDomain = _1;

    template <typename CellShapeTag>
    VISKORES_EXEC void operator()(CellShapeTag shape, viskores::IdComponent& numFaces) const
    {
      viskores::exec::CellFaceNumberOfFaces(shape, numFaces);
    }
  };

  // Worklet that identifies a cell face by a hash value. Not necessarily completely unique.
  class FaceHash : public viskores::worklet::WorkletVisitCellsWithPoints
  {
  public:
    using ControlSignature = void(
      CellSetIn cellset, FieldOut faceHashes, FieldOut originCells, FieldOut originFaces);
    using ExecutionSignature = void(_2, _3, _4, CellShape, PointIndices, InputIndex, VisitIndex);
    using InputDomain = _1;

    using ScatterType = viskores::worklet::ScatterCounting;

    template <typename CellShapeTag, typename CellNodeVecType>
    VISKORES_EXEC void operator()(viskores::HashType& faceHash, viskores::Id& cellIndex,
      viskores::IdComponent& faceIndex, CellShapeTag shape, const CellNodeVecType& cellNodeIds,
      viskores::Id inputIndex, viskores::IdComponent visitIndex) const
    {
      viskores::Id3 faceId;
      viskores::exec::CellFaceCanonicalId(visitIndex, shape, cellNodeIds, faceId);
      faceHash = viskores::Hash(faceId);

      cellIndex = inputIndex;
      faceIndex = visitIndex;
    }
  };

  // Worklet that identifies the number of cells written out per face.
  // Because there can be collisions in the face ids, this instance might
  // represent multiple faces, which have to be checked. The resulting
  // number is the total number of external faces.
  class FaceCounts : public viskores::worklet::WorkletReduceByKey
  {
  public:
    using ControlSignature = void(KeysIn keys, WholeCellSetIn<> inputCells, ValuesIn originCells,
      ValuesIn originFaces, ReducedValuesOut numOutputCells);
    using ExecutionSignature = _5(_2, _3, _4);
    using InputDomain = _1;

    template <typename CellSetType, typename OriginCellsType, typename OriginFacesType>
    VISKORES_EXEC viskores::IdComponent operator()(const CellSetType& cellSet,
      const OriginCellsType& originCells, const OriginFacesType& originFaces) const
    {
      viskores::IdComponent numCellsOnHash = originCells.GetNumberOfComponents();
      VISKORES_ASSERT(originFaces.GetNumberOfComponents() == numCellsOnHash);

      // Start by assuming all faces are unique, then remove one for each
      // face we find a duplicate for.
      viskores::IdComponent numExternalFaces = numCellsOnHash;

      for (viskores::IdComponent myIndex = 0;
        myIndex < numCellsOnHash - 1; // Don't need to check last face
        myIndex++)
      {
        viskores::Id3 myFace;
        viskores::exec::CellFaceCanonicalId(originFaces[myIndex],
          cellSet.GetCellShape(originCells[myIndex]), cellSet.GetIndices(originCells[myIndex]),
          myFace);
        for (viskores::IdComponent otherIndex = myIndex + 1; otherIndex < numCellsOnHash;
          otherIndex++)
        {
          viskores::Id3 otherFace;
          viskores::exec::CellFaceCanonicalId(originFaces[otherIndex],
            cellSet.GetCellShape(originCells[otherIndex]),
            cellSet.GetIndices(originCells[otherIndex]), otherFace);
          if (myFace == otherFace)
          {
            // Faces are the same. Must be internal. Remove 2, one for each face. We don't have to
            // worry about otherFace matching anything else because a proper topology will have at
            // most 2 cells sharing a face, so there should be no more matches.
            numExternalFaces -= 2;
            break;
          }
        }
      }

      return numExternalFaces;
    }
  };

private:
  // Resolves duplicate hashes by finding a specified unique face for a given hash.
  // Given a cell set (from a WholeCellSetIn) and the cell/face id pairs for each face
  // associated with a given hash, returns the index of the cell/face provided of the
  // visitIndex-th unique face. Basically, this method searches through all the cell/face
  // pairs looking for unique sets and returns the one associated with visitIndex.
  template <typename CellSetType, typename OriginCellsType, typename OriginFacesType>
  VISKORES_EXEC static viskores::IdComponent FindUniqueFace(const CellSetType& cellSet,
    const OriginCellsType& originCells, const OriginFacesType& originFaces,
    viskores::IdComponent visitIndex)
  {
    viskores::IdComponent numCellsOnHash = originCells.GetNumberOfComponents();
    VISKORES_ASSERT(originFaces.GetNumberOfComponents() == numCellsOnHash);

    // Find the visitIndex-th unique face.
    viskores::IdComponent numFound = 0;
    viskores::IdComponent myIndex = 0;
    while (true)
    {
      VISKORES_ASSERT(myIndex < numCellsOnHash);
      viskores::Id3 myFace;
      viskores::exec::CellFaceCanonicalId(originFaces[myIndex],
        cellSet.GetCellShape(originCells[myIndex]), cellSet.GetIndices(originCells[myIndex]),
        myFace);
      bool foundPair = false;
      for (viskores::IdComponent otherIndex = 0; otherIndex < numCellsOnHash; otherIndex++)
      {
        if (otherIndex == myIndex)
        {
          continue;
        }
        viskores::Id3 otherFace;
        viskores::exec::CellFaceCanonicalId(originFaces[otherIndex],
          cellSet.GetCellShape(originCells[otherIndex]),
          cellSet.GetIndices(originCells[otherIndex]), otherFace);
        if (myFace == otherFace)
        {
          // Faces are the same. Must be internal.
          foundPair = true;
          break;
        }
      }

      if (!foundPair)
      {
        if (numFound == visitIndex)
        {
          break;
        }
        else
        {
          numFound++;
        }
      }

      myIndex++;
    }

    return myIndex;
  }

public:
  // Worklet that returns the number of points for each outputted face.
  // Have to manage the case where multiple faces have the same hash.
  class NumPointsPerFace : public viskores::worklet::WorkletReduceByKey
  {
  public:
    using ControlSignature = void(KeysIn keys, WholeCellSetIn<> inputCells, ValuesIn originCells,
      ValuesIn originFaces, ReducedValuesOut numPointsInFace);
    using ExecutionSignature = void(_2, _3, _4, VisitIndex, _5);
    using InputDomain = _1;

    using ScatterType = viskores::worklet::ScatterCounting;

    template <typename CountArrayType>
    VISKORES_CONT static ScatterType MakeScatter(const CountArrayType& countArray)
    {
      VISKORES_IS_ARRAY_HANDLE(CountArrayType);
      return ScatterType(countArray);
    }

    template <typename CellSetType, typename OriginCellsType, typename OriginFacesType>
    VISKORES_EXEC void operator()(const CellSetType& cellSet, const OriginCellsType& originCells,
      const OriginFacesType& originFaces, viskores::IdComponent visitIndex,
      viskores::IdComponent& numFacePoints) const
    {
      viskores::IdComponent myIndex =
        ExternalFacesHashSortFnv1a::FindUniqueFace(cellSet, originCells, originFaces, visitIndex);

      viskores::exec::CellFaceNumberOfPoints(
        originFaces[myIndex], cellSet.GetCellShape(originCells[myIndex]), numFacePoints);
    }
  };

  // Worklet that returns the shape and connectivity for each external face
  class BuildConnectivity : public viskores::worklet::WorkletReduceByKey
  {
  public:
    using ControlSignature = void(KeysIn keys, WholeCellSetIn<> inputCells, ValuesIn originCells,
      ValuesIn originFaces, ReducedValuesOut shapesOut, ReducedValuesOut connectivityOut,
      ReducedValuesOut cellIdMapOut);
    using ExecutionSignature = void(_2, _3, _4, VisitIndex, _5, _6, _7);
    using InputDomain = _1;

    using ScatterType = viskores::worklet::ScatterCounting;

    template <typename CellSetType, typename OriginCellsType, typename OriginFacesType,
      typename ConnectivityType>
    VISKORES_EXEC void operator()(const CellSetType& cellSet, const OriginCellsType& originCells,
      const OriginFacesType& originFaces, viskores::IdComponent visitIndex,
      viskores::UInt8& shapeOut, ConnectivityType& connectivityOut,
      viskores::Id& cellIdMapOut) const
    {
      const viskores::IdComponent myIndex =
        ExternalFacesHashSortFnv1a::FindUniqueFace(cellSet, originCells, originFaces, visitIndex);
      const viskores::IdComponent myFace = originFaces[myIndex];

      typename CellSetType::CellShapeTag shapeIn = cellSet.GetCellShape(originCells[myIndex]);
      viskores::exec::CellFaceShape(myFace, shapeIn, shapeOut);
      cellIdMapOut = originCells[myIndex];

      viskores::IdComponent numFacePoints;
      viskores::exec::CellFaceNumberOfPoints(myFace, shapeIn, numFacePoints);

      VISKORES_ASSERT(numFacePoints == connectivityOut.GetNumberOfComponents());

      typename CellSetType::IndicesType inCellIndices = cellSet.GetIndices(originCells[myIndex]);

      for (viskores::IdComponent facePointIndex = 0; facePointIndex < numFacePoints;
        facePointIndex++)
      {
        viskores::IdComponent localFaceIndex;
        viskores::ErrorCode status =
          viskores::exec::CellFaceLocalIndex(facePointIndex, myFace, shapeIn, localFaceIndex);
        if (status == viskores::ErrorCode::Success)
        {
          connectivityOut[facePointIndex] = inCellIndices[localFaceIndex];
        }
        else
        {
          // An error condition, but do we want to crash the operation?
          connectivityOut[facePointIndex] = 0;
        }
      }
    }
  };

public:
  VISKORES_CONT
  ExternalFacesHashSortFnv1a() {}

  void ReleaseCellMapArrays() { this->CellIdMap.ReleaseResources(); }

  ///////////////////////////////////////////////////
  /// \brief ExternalFaces: Extract Faces on outside of geometry
  template <typename InCellSetType, typename ShapeStorage, typename ConnectivityStorage,
    typename OffsetsStorage>
  VISKORES_CONT void Run(const InCellSetType& inCellSet,
    viskores::cont::CellSetExplicit<ShapeStorage, ConnectivityStorage, OffsetsStorage>& outCellSet,
    YamlWriter& log)
  {
    using PointCountArrayType = viskores::cont::ArrayHandle<viskores::IdComponent>;
    using ShapeArrayType = viskores::cont::ArrayHandle<viskores::UInt8, ShapeStorage>;
    using OffsetsArrayType = viskores::cont::ArrayHandle<viskores::Id, OffsetsStorage>;
    using ConnectivityArrayType = viskores::cont::ArrayHandle<viskores::Id, ConnectivityStorage>;

    // Create a worklet to map the number of faces to each cell
    viskores::cont::ArrayHandle<viskores::IdComponent> facesPerCell;
    viskores::worklet::DispatcherMapTopology<NumFacesPerCell> numFacesDispatcher;

    viskores::cont::Timer timer;
    timer.Start();
    numFacesDispatcher.Invoke(inCellSet, facesPerCell);
    timer.Stop();
    log.AddDictionaryEntry("seconds-num-faces-per-cell", timer.GetElapsedTime());

    timer.Start();
    viskores::worklet::ScatterCounting scatterCellToFace(facesPerCell);
    timer.Stop();
    log.AddDictionaryEntry("seconds-face-input-count", timer.GetElapsedTime());
    facesPerCell.ReleaseResources();

    if (scatterCellToFace.GetOutputRange(inCellSet.GetNumberOfCells()) == 0)
    {
      // Data has no faces. Output is empty.
      outCellSet.PrepareToAddCells(0, 0);
      outCellSet.CompleteAddingCells(inCellSet.GetNumberOfPoints());
      return;
    }

    viskores::cont::ArrayHandle<viskores::HashType> faceHashes;
    viskores::cont::ArrayHandle<viskores::Id> originCells;
    viskores::cont::ArrayHandle<viskores::IdComponent> originFaces;
    viskores::worklet::DispatcherMapTopology<FaceHash> faceHashDispatcher(scatterCellToFace);

    timer.Start();
    faceHashDispatcher.Invoke(inCellSet, faceHashes, originCells, originFaces);
    timer.Stop();
    log.AddDictionaryEntry("seconds-face-hash", timer.GetElapsedTime());

    timer.Start();
    viskores::worklet::Keys<viskores::HashType> faceKeys(faceHashes);
    timer.Stop();
    log.AddDictionaryEntry("seconds-keys-build-arrays", timer.GetElapsedTime());

    viskores::cont::ArrayHandle<viskores::IdComponent> faceOutputCount;
    viskores::worklet::DispatcherReduceByKey<FaceCounts> faceCountDispatcher;

    timer.Start();
    faceCountDispatcher.Invoke(faceKeys, inCellSet, originCells, originFaces, faceOutputCount);
    timer.Stop();
    log.AddDictionaryEntry("seconds-face-count", timer.GetElapsedTime());

    timer.Start();
    auto scatterCullInternalFaces = NumPointsPerFace::MakeScatter(faceOutputCount);
    timer.Stop();
    log.AddDictionaryEntry("seconds-face-output-count", timer.GetElapsedTime());

    PointCountArrayType facePointCount;
    viskores::worklet::DispatcherReduceByKey<NumPointsPerFace> pointsPerFaceDispatcher(
      scatterCullInternalFaces);

    timer.Start();
    pointsPerFaceDispatcher.Invoke(faceKeys, inCellSet, originCells, originFaces, facePointCount);
    timer.Stop();
    log.AddDictionaryEntry("seconds-points-per-face", timer.GetElapsedTime());

    ShapeArrayType faceShapes;

    OffsetsArrayType faceOffsets;
    viskores::Id connectivitySize;
    timer.Start();
    viskores::cont::ConvertNumComponentsToOffsets(facePointCount, faceOffsets, connectivitySize);
    timer.Stop();
    log.AddDictionaryEntry("seconds-face-point-count", timer.GetElapsedTime());

    ConnectivityArrayType faceConnectivity;
    // Must pre allocate because worklet invocation will not have enough
    // information to.
    faceConnectivity.Allocate(connectivitySize);

    viskores::worklet::DispatcherReduceByKey<BuildConnectivity> buildConnectivityDispatcher(
      scatterCullInternalFaces);

    viskores::cont::ArrayHandle<viskores::Id> faceToCellIdMap;

    timer.Start();
    buildConnectivityDispatcher.Invoke(faceKeys, inCellSet, originCells, originFaces, faceShapes,
      viskores::cont::make_ArrayHandleGroupVecVariable(faceConnectivity, faceOffsets),
      faceToCellIdMap);
    timer.Stop();
    log.AddDictionaryEntry("seconds-build-connectivity", timer.GetElapsedTime());

    outCellSet.Fill(inCellSet.GetNumberOfPoints(), faceShapes, faceConnectivity, faceOffsets);
    this->CellIdMap = faceToCellIdMap;
  }

  viskores::cont::ArrayHandle<viskores::Id> GetCellIdMap() const { return this->CellIdMap; }

private:
  viskores::cont::ArrayHandle<viskores::Id> CellIdMap;

}; // struct ExternalFaces
}
} // namespace viskores::worklet

#endif // viskores_worklet_ExternalFacesHashSortFnv1a_h
