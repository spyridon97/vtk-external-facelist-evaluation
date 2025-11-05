//============================================================================
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//============================================================================
#ifndef viskores_worklet_ExternalFacesHashFightMinPointId_h
#define viskores_worklet_ExternalFacesHashFightMinPointId_h

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

#include "viskores/worklet/WorkletMapField.h"
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

struct ExternalFacesHashFightMinPointId
{
  // Unary predicate operator
  // Returns True if the argument is equal to the constructor
  // integer argument; False otherwise.
  struct IsIntValue
  {
  private:
    int Value;

  public:
    VISKORES_EXEC_CONT
    IsIntValue(const int& v)
      : Value(v)
    {
    }

    template <typename T>
    VISKORES_EXEC_CONT bool operator()(const T& x) const
    {
      return x == T(Value);
    }
  };

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
      viskores::Id minFacePointId;
      viskores::exec::CellFaceMinPointId(visitIndex, shape, cellNodeIds, minFacePointId);
      faceHash = static_cast<viskores::HashType>(minFacePointId);

      cellIndex = inputIndex;
      faceIndex = visitIndex;
    }
  };

  // Worklet that writes the face index at the location of the hash table.
  // Multiple entries are likely to write to the hash table, so they fight
  // and (hopefully) one wins.
  class HashFight : public viskores::worklet::WorkletMapField
  {
  public:
    using ControlSignature = void(FieldIn Hashes, FieldIn FaceIds, WholeArrayOut HashTable);
    typedef void ExecutionSignature(_1, _2, _3);

    VISKORES_CONT
    HashFight(viskores::Id hashTableSize)
      : HashTableSize(hashTableSize)
    {
    }

    template <typename HashTablePortalType>
    VISKORES_EXEC void operator()(
      viskores::Id hash, viskores::Id faceId, HashTablePortalType& hashTablePortal) const
    {
      hashTablePortal.Set(hash % this->HashTableSize, faceId);
    }

  private:
    viskores::Id HashTableSize;
  };

  // Worklet that detects whether a face is internal.  If the
  // face is internal, then a value should not be assigned to the
  // face in the output array handle of face vertices; only external
  // faces should have a vector not equal to <-1,-1,-1>
  class CheckForMatches : public viskores::worklet::WorkletMapField
  {
  public:
    typedef void ControlSignature(FieldIn activeHashes, FieldIn activeFaceIndices,
      WholeCellSetIn<> cellSet, WholeArrayIn originCells, WholeArrayIn originFaces,
      WholeArrayIn hashTable, FieldInOut isInactive, WholeArrayInOut isExternalFace);
    typedef void ExecutionSignature(_1, _2, _3, _4, _5, _6, _7, _8);

    VISKORES_CONT
    CheckForMatches(viskores::Id hashTableSize)
      : HashTableSize(hashTableSize)
    {
    }

    template <typename CellSetType, typename OriginCellsPortal, typename OriginFacesPortal,
      typename HashTablePortal, typename IsExternalFacePortal>
    VISKORES_EXEC void operator()(viskores::UInt32 hash, viskores::Id faceIndex,
      const CellSetType& cellSet, const OriginCellsPortal& originCellsPortal,
      const OriginFacesPortal& originFacesPortal, const HashTablePortal& hashTablePortal,
      viskores::UInt8& isInactive, IsExternalFacePortal& isExternalFacePortal) const
    {
      viskores::Id hashWinnerFace = hashTablePortal.Get(hash % this->HashTableSize);

      if (hashWinnerFace == faceIndex)
      {
        // Case 1: I won the hash fight by writing my index. I'm done so mark
        // myself as inactive.
        isInactive = viskores::UInt8(1);
      }
      else
      {
        // Get a cononical representation of my face.
        viskores::Id myOriginCell = originCellsPortal.Get(faceIndex);
        viskores::IdComponent myOriginFace = originFacesPortal.Get(faceIndex);
        viskores::Id3 myFace;
        viskores::exec::CellFaceCanonicalId(myOriginFace, cellSet.GetCellShape(myOriginCell),
          cellSet.GetIndices(myOriginCell), myFace);

        // Get a cononical representation of the face in the hash table.
        viskores::Id otherOriginCell = originCellsPortal.Get(hashWinnerFace);
        viskores::IdComponent otherOriginFace = originFacesPortal.Get(hashWinnerFace);
        viskores::Id3 otherFace;
        viskores::exec::CellFaceCanonicalId(otherOriginFace, cellSet.GetCellShape(otherOriginCell),
          cellSet.GetIndices(otherOriginCell), otherFace);

        // See if these are the same face
        if (/*myFace[0] == otherFace[0] && */ myFace[1] == otherFace[1] &&
          myFace[2] == otherFace[2])
        {
          // Case 2: The faces are the same. This must be an internal face.
          // Mark both myself and the other face as internal.
          isInactive = viskores::UInt8(1);
          isExternalFacePortal.Set(faceIndex, viskores::UInt8(0));
          isExternalFacePortal.Set(hashWinnerFace, viskores::UInt8(0));
        }
        else
        {
          // Case 3: I didn't win and my face didn't match. I didn't learn
          // anything so do nothing.
        }
      }
    }

  private:
    viskores::Id HashTableSize;
  };

  // Worklet that counts the number of points that are in each (active) face.
  class NumPointsPerFace : public viskores::worklet::WorkletMapField
  {
  public:
    typedef void ControlSignature(FieldIn faceIndices, WholeCellSetIn<> cellSet,
      WholeArrayIn originCells, WholeArrayIn originFaces, FieldOut numPointsInFace);
    typedef _5 ExecutionSignature(_1, _2, _3, _4);

    using ScatterType = viskores::worklet::ScatterCounting;

    template <typename CellSetType, typename OriginCellsPortalType, typename OriginFacesPortalType>
    VISKORES_EXEC viskores::IdComponent operator()(viskores::Id faceIndex,
      const CellSetType& cellSet, const OriginCellsPortalType& originCellsPortal,
      const OriginFacesPortalType& originFacesPortal) const
    {
      viskores::Id originCell = originCellsPortal.Get(faceIndex);
      viskores::IdComponent originFace = originFacesPortal.Get(faceIndex);

      viskores::IdComponent numFacePoints;
      viskores::exec::CellFaceNumberOfPoints(
        originFace, cellSet.GetCellShape(originCell), numFacePoints);
      return numFacePoints;
    }
  };

  // Worklet that writes out the shape and indices for each (active) face.
  class BuildConnectivity : public viskores::worklet::WorkletMapField
  {
  public:
    typedef void ControlSignature(FieldIn faceIndices, WholeCellSetIn<> cellSet,
      WholeArrayIn originCells, WholeArrayIn originFaces, FieldOut shapesOut,
      FieldOut connectivityOut, FieldOut cellIdMapOut);
    typedef void ExecutionSignature(_1, _2, _3, _4, _5, _6, _7);

    using ScatterType = viskores::worklet::ScatterCounting;

    template <typename CellSetType, typename OriginCellsPortalType, typename OriginFacesPortalType,
      typename ConnectivityType>
    VISKORES_EXEC void operator()(const viskores::Id faceIndex, const CellSetType& cellSet,
      const OriginCellsPortalType& originCellsPortal,
      const OriginFacesPortalType& originFacesPortal, viskores::UInt8& shapeOut,
      ConnectivityType& connectivityOut, viskores::Id& cellIdMapOut) const
    {
      viskores::Id originCell = originCellsPortal.Get(faceIndex);
      viskores::IdComponent originFace = originFacesPortal.Get(faceIndex);

      viskores::exec::CellFaceShape(originFace, cellSet.GetCellShape(originCell), shapeOut);
      cellIdMapOut = originCell;

      viskores::IdComponent numFacePoints;
      viskores::exec::CellFaceNumberOfPoints(
        originFace, cellSet.GetCellShape(originCell), numFacePoints);
      VISKORES_ASSERT(numFacePoints == connectivityOut.GetNumberOfComponents());

      typename CellSetType::IndicesType inCellIndices = cellSet.GetIndices(originCell);

      for (viskores::IdComponent facePointIndex = 0; facePointIndex < numFacePoints;
        facePointIndex++)
      {
        viskores::IdComponent localFaceIndex;
        viskores::ErrorCode status = viskores::exec::CellFaceLocalIndex(
          facePointIndex, originFace, cellSet.GetCellShape(originCell), localFaceIndex);
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
  ExternalFacesHashFightMinPointId() {}

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

    viskores::Id totalNumFaces = faceHashes.GetNumberOfValues();

    timer.Start();

    // Set constant factor for hash table size: factor*totalFaces
    const viskores::Id hashTableFactor = 2;

    viskores::cont::ArrayHandle<viskores::UInt8> isExternalFace;
    viskores::cont::Algorithm::Copy(
      viskores::cont::ArrayHandleConstant<viskores::UInt8>(1, totalNumFaces), isExternalFace);

    viskores::cont::ArrayHandle<viskores::Id> activeFaceIndices;
    viskores::cont::Algorithm::Copy(
      viskores::cont::ArrayHandleIndex(totalNumFaces), activeFaceIndices);

    viskores::Id numActiveFaces = totalNumFaces;

    while (numActiveFaces > 0)
    {
      // Create a packe arrays of active face hashes
      auto activeHashes =
        viskores::cont::make_ArrayHandlePermutation(activeFaceIndices, faceHashes);

      // Get ready the isInactive array.
      viskores::cont::ArrayHandle<viskores::UInt8> isInactive;
      viskores::cont::Algorithm::Copy(
        viskores::cont::ArrayHandleConstant<viskores::UInt8>(0, numActiveFaces), isInactive);

      viskores::Id hashTableSize = numActiveFaces * hashTableFactor;

      viskores::cont::ArrayHandle<viskores::Id> hashTable;
      hashTable.Allocate(hashTableSize);

      // Have all active hashes try to write their index to the hash table
      viskores::worklet::DispatcherMapField<HashFight> fightDispatcher((HashFight(hashTableSize)));
      fightDispatcher.Invoke(activeHashes, activeFaceIndices, hashTable);

      // Have all active faces check to see if they matched and update
      // isInactive/isExternalFace.
      viskores::worklet::DispatcherMapField<CheckForMatches> matchDispatcher(
        (CheckForMatches(hashTableSize)));
      matchDispatcher.Invoke(activeHashes, activeFaceIndices, inCellSet, originCells, originFaces,
        hashTable, isInactive, isExternalFace);

      // Compact the activeFaceIndices by the isInactive flag.
      viskores::cont::ArrayHandle<viskores::Id> compactedActiveFaceIndices;
      viskores::cont::Algorithm::CopyIf(
        activeFaceIndices, isInactive, compactedActiveFaceIndices, IsIntValue(0));
      activeFaceIndices = compactedActiveFaceIndices;

      // Update the number of active faces
      numActiveFaces = activeFaceIndices.GetNumberOfValues();
    }
    timer.Stop();
    log.AddDictionaryEntry("seconds-hash-fight-iterations", timer.GetElapsedTime());

    viskores::worklet::ScatterCounting scatterCullInternalFaces(isExternalFace);

    PointCountArrayType facePointCount;
    viskores::worklet::DispatcherMapField<NumPointsPerFace> pointsPerFaceDispatcher(
      scatterCullInternalFaces);

    timer.Start();
    pointsPerFaceDispatcher.Invoke(viskores::cont::ArrayHandleIndex(totalNumFaces), inCellSet,
      originCells, originFaces, facePointCount);
    timer.Stop();
    log.AddDictionaryEntry("seconds-face-output-count", timer.GetElapsedTime());

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

    viskores::worklet::DispatcherMapField<BuildConnectivity> buildConnectivityDispatcher(
      scatterCullInternalFaces);

    viskores::cont::ArrayHandle<viskores::Id> faceToCellIdMap;

    // Create a view that doesn't have the last offset:
    auto faceOffsetsTrim =
      viskores::cont::make_ArrayHandleView(faceOffsets, 0, faceOffsets.GetNumberOfValues() - 1);

    timer.Start();
    buildConnectivityDispatcher.Invoke(viskores::cont::ArrayHandleIndex(totalNumFaces), inCellSet,
      originCells, originFaces, faceShapes,
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

}; // struct ExternalFacesHashFightMinPointId
}
} // namespace viskores::worklet

#endif // viskores_worklet_ExternalFacesHashFightMinPointId_h
