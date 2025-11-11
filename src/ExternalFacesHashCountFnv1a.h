//============================================================================
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//============================================================================
#ifndef viskores_worklet_ExternalFacesHashCountFnv1a_h
#define viskores_worklet_ExternalFacesHashCountFnv1a_h

#include <viskores/CellShape.h>
#include <viskores/Hash.h>
#include <viskores/Math.h>
#include <viskores/Swap.h>

#include <viskores/exec/CellFace.h>

#include <viskores/cont/Algorithm.h>
#include <viskores/cont/ArrayHandle.h>
#include <viskores/cont/ArrayHandleGroupVecVariable.h>
#include <viskores/cont/CellSetExplicit.h>
#include <viskores/cont/ConvertNumComponentsToOffsets.h>
#include <viskores/cont/Timer.h>

#include <viskores/worklet/ScatterCounting.h>
#include <viskores/worklet/WorkletMapField.h>
#include <viskores/worklet/WorkletMapTopology.h>

#include "YamlWriter.h"

namespace viskores
{
namespace worklet
{

struct ExternalFacesHashCountFnv1a
{
  // Worklet that returns the number of faces for each cell/shape
  class NumFacesPerCell : public viskores::worklet::WorkletVisitCellsWithPoints
  {
  public:
    using ControlSignature = void(CellSetIn inCellSet, FieldOut numFacesInCell);
    using ExecutionSignature = void(CellShape, _2);
    using InputDomain = _1;

    template <typename CellShapeTag>
    VISKORES_EXEC void operator()(CellShapeTag shape, viskores::IdComponent& numFacesInCell) const
    {
      viskores::exec::CellFaceNumberOfFaces(shape, numFacesInCell);
    }
  };

  // Worklet that identifies a cell face by a hash value. Not necessarily completely unique.
  class FaceHash : public viskores::worklet::WorkletVisitCellsWithPoints
  {
  public:
    using ControlSignature = void(CellSetIn cellset, FieldOutCell cellFaceHashes);
    using ExecutionSignature = void(CellShape, PointIndices, _2);
    using InputDomain = _1;

    explicit FaceHash(const viskores::Id& hashTableSize)
      : HashTableSize(hashTableSize)
    {
    }

    template <typename CellShapeTag, typename CellNodeVecType, typename CellFaceHashes>
    VISKORES_EXEC void operator()(const CellShapeTag shape, const CellNodeVecType& cellNodeIds,
      CellFaceHashes& cellFaceHashes) const
    {
      const viskores::IdComponent numFaces = cellFaceHashes.GetNumberOfComponents();
      for (viskores::IdComponent faceIndex = 0; faceIndex < numFaces; ++faceIndex)
      {
        viskores::Id3 faceId;
        viskores::exec::CellFaceCanonicalId(faceIndex, shape, cellNodeIds, faceId);
        cellFaceHashes[faceIndex] = (viskores::Hash(faceId) % this->HashTableSize);
      }
    }

  private:
    viskores::Id HashTableSize;
  };

  // Worklet that identifies the number of faces per hash.
  class NumFacesPerHash : public viskores::worklet::WorkletMapField
  {
  public:
    using ControlSignature = void(FieldIn faceHashes, AtomicArrayInOut numFacesPerHash);
    using ExecutionSignature = void(_1, _2);
    using InputDomain = _1;

    template <typename NumFacesPerHashArray>
    VISKORES_EXEC void operator()(
      const viskores::HashType& faceHash, NumFacesPerHashArray& numFacesPerHash) const
    {
      // MemoryOrder::Relaxed is safe here, since we're not using the atomics for synchronization.
      numFacesPerHash.Add(faceHash, 1, viskores::MemoryOrder::Relaxed);
    }
  };

  /// Class to pack and unpack cell and face indices to/from a single integer.
  class CellFaceIdPacker
  {
  public:
    using CellAndFaceIdType = viskores::UInt64;
    using CellIdType = viskores::Id;
    using FaceIdType = viskores::Int8;

    static constexpr CellAndFaceIdType GetNumFaceIdBits()
    {
      static_assert(viskores::exec::detail::CellFaceTables::MAX_NUM_FACES == 6,
        "MAX_NUM_FACES must be 6, otherwise, update GetNumFaceIdBits");
      return 3;
    }
    static constexpr CellAndFaceIdType GetFaceMask() { return (1ULL << GetNumFaceIdBits()) - 1; }

    /// Pack function for both cellIndex and faceIndex
    VISKORES_EXEC inline static constexpr CellAndFaceIdType Pack(
      const CellIdType& cellIndex, const FaceIdType& faceIndex)
    {
      // Pack the cellIndex in the higher bits, leaving FACE_INDEX_BITS bits for faceIndex
      return static_cast<CellAndFaceIdType>(cellIndex << GetNumFaceIdBits()) |
        static_cast<CellAndFaceIdType>(faceIndex);
    }

    /// Unpacking function for both cellIndex and faceIndex
    /// This is templated because we don't want to create a copy of the packedCellAndFaceId value.
    template <typename TCellAndFaceIdType>
    VISKORES_EXEC inline static constexpr void Unpack(
      const TCellAndFaceIdType& packedCellAndFaceId, CellIdType& cellIndex, FaceIdType& faceIndex)
    {
      // Extract faceIndex from the lower GetNumFaceIdBits bits
      faceIndex = static_cast<FaceIdType>(packedCellAndFaceId & GetFaceMask());
      // Extract cellIndex by shifting back
      cellIndex = static_cast<CellIdType>(packedCellAndFaceId >> GetNumFaceIdBits());
    }
  };

  // Worklet that writes out the cell and face ids of each face per hash.
  class BuildFacesPerHash : public viskores::worklet::WorkletMapField
  {
  public:
    using ControlSignature = void(FieldIn cellFaceHashes, AtomicArrayInOut numFacesPerHash,
      WholeArrayOut cellAndFaceIdOfFacesPerHash);
    using ExecutionSignature = void(InputIndex, _1, _2, _3);
    using InputDomain = _1;

    template <typename CellFaceHashes, typename NumFacesPerHashArray,
      typename CellAndFaceIdOfFacePerHashArray>
    VISKORES_EXEC void operator()(viskores::Id inputIndex, const CellFaceHashes& cellFaceHashes,
      NumFacesPerHashArray& numFacesPerHash,
      CellAndFaceIdOfFacePerHashArray& cellAndFaceIdOfFacesPerHash) const
    {
      const viskores::IdComponent numFaces = cellFaceHashes.GetNumberOfComponents();
      for (viskores::IdComponent faceIndex = 0; faceIndex < numFaces; ++faceIndex)
      {
        const auto& faceHash = cellFaceHashes[faceIndex];
        // MemoryOrder::Relaxed is safe here, since we're not using the atomics for synchronization.
        const viskores::IdComponent hashFaceIndex =
          numFacesPerHash.Add(faceHash, -1, viskores::MemoryOrder::Relaxed) - 1;
        cellAndFaceIdOfFacesPerHash.Get(faceHash)[hashFaceIndex] =
          CellFaceIdPacker::Pack(inputIndex, static_cast<CellFaceIdPacker::FaceIdType>(faceIndex));
      }
    }
  };

  // Worklet that identifies the number of external faces per Hash.
  // Because there can be collisions in the hash, this instance hash might
  // represent multiple faces, which have to be checked. The resulting
  // number is the total number of external faces. It also reorders the
  // faces so that the external faces are first, followed by the internal faces.
  class FaceCounts : public viskores::worklet::WorkletMapField
  {
  public:
    using ControlSignature = void(FieldInOut cellAndFaceIdOfFacesInHash,
      WholeCellSetIn<> inputCells, FieldOut externalFacesInHash);
    using ExecutionSignature = _3(_1, _2);
    using InputDomain = _1;

    template <typename CellAndFaceIdOfFacesInHash, typename CellSetType>
    VISKORES_EXEC viskores::IdComponent operator()(
      CellAndFaceIdOfFacesInHash& cellAndFaceIdOfFacesInHash, const CellSetType& cellSet) const
    {
      const viskores::IdComponent numFacesInHash =
        cellAndFaceIdOfFacesInHash.GetNumberOfComponents();

      static constexpr viskores::IdComponent FACE_CANONICAL_IDS_CACHE_SIZE = 100;
      if (numFacesInHash <= 1)
      {
        // Either one or zero faces. If there is one, it's external, In either case, do nothing.
        return numFacesInHash;
      }
      else if (numFacesInHash <= FACE_CANONICAL_IDS_CACHE_SIZE) // Fast path with caching
      {
        CellFaceIdPacker::CellIdType myCellId;
        CellFaceIdPacker::FaceIdType myFaceId;
        viskores::Vec<viskores::Id3, FACE_CANONICAL_IDS_CACHE_SIZE> faceCanonicalIds;
        for (viskores::IdComponent faceIndex = 0; faceIndex < numFacesInHash; ++faceIndex)
        {
          CellFaceIdPacker::Unpack(cellAndFaceIdOfFacesInHash[faceIndex], myCellId, myFaceId);
          viskores::exec::CellFaceCanonicalId(myFaceId, cellSet.GetCellShape(myCellId),
            cellSet.GetIndices(myCellId), faceCanonicalIds[faceIndex]);
        }
        // Start by assuming all faces are duplicate, then remove two for each duplicate pair found.
        viskores::IdComponent numExternalFaces = 0;
        // Iterate over the faces in the hash in reverse order (to minimize the swaps being
        // performed) and find duplicates faces. Put duplicates at the end and unique faces
        // at the beginning. Narrow this range until all unique/duplicate are found.
        for (viskores::IdComponent myIndex = numFacesInHash - 1; myIndex >= numExternalFaces;)
        {
          bool isInternal = false;
          const viskores::Id3& myFace = faceCanonicalIds[myIndex];
          viskores::IdComponent otherIndex;
          for (otherIndex = myIndex - 1; otherIndex >= numExternalFaces; --otherIndex)
          {
            const viskores::Id3& otherFace = faceCanonicalIds[otherIndex];
            if (myFace[0] == otherFace[0] && myFace[1] == otherFace[1] && myFace[2] == otherFace[2])
            {
              // Faces are the same. Must be internal. We don't have to worry about otherFace
              // matching anything else because a proper topology will have at most 2 cells sharing
              // a face, so there should be no more matches.
              isInternal = true;
              break;
            }
          }
          if (isInternal) // If two faces are internal,
          {               // swap them to the end of the list to avoid revisiting them.
            --myIndex;    // decrement for the first duplicate face, which is at the end
            if (myIndex != otherIndex)
            {
              FaceCounts::SwapFace<CellFaceIdPacker::CellAndFaceIdType>(
                cellAndFaceIdOfFacesInHash[otherIndex], cellAndFaceIdOfFacesInHash[myIndex]);
              viskores::Swap(faceCanonicalIds[otherIndex], faceCanonicalIds[myIndex]);
            }
            --myIndex; // decrement for the second duplicate face
          }
          else // If the face is external
          {    // swap it to the front of the list, to avoid revisiting it.
            if (myIndex != numExternalFaces)
            {
              FaceCounts::SwapFace<CellFaceIdPacker::CellAndFaceIdType>(
                cellAndFaceIdOfFacesInHash[myIndex], cellAndFaceIdOfFacesInHash[numExternalFaces]);
              viskores::Swap(faceCanonicalIds[myIndex], faceCanonicalIds[numExternalFaces]);
            }
            ++numExternalFaces; // increment for the new external face
            // myIndex remains the same, since we have a new face to check at the same myIndex.
            // However, numExternalFaces has incremented, so the loop could still terminate.
          }
        }
        return numExternalFaces;
      }
      else // Slow path without caching
      {
        CellFaceIdPacker::CellIdType myCellId, otherCellId;
        CellFaceIdPacker::FaceIdType myFaceId, otherFaceId;
        viskores::Id3 myFace, otherFace;
        // Start by assuming all faces are duplicate, then remove two for each duplicate pair found.
        viskores::IdComponent numExternalFaces = 0;
        // Iterate over the faces in the hash in reverse order (to minimize the swaps being
        // performed) and find duplicates faces. Put duplicates at the end and unique faces
        // at the beginning. Narrow this range until all unique/duplicate are found.
        for (viskores::IdComponent myIndex = numFacesInHash - 1; myIndex >= numExternalFaces;)
        {
          bool isInternal = false;
          CellFaceIdPacker::Unpack(cellAndFaceIdOfFacesInHash[myIndex], myCellId, myFaceId);
          viskores::exec::CellFaceCanonicalId(
            myFaceId, cellSet.GetCellShape(myCellId), cellSet.GetIndices(myCellId), myFace);
          viskores::IdComponent otherIndex;
          for (otherIndex = myIndex - 1; otherIndex >= numExternalFaces; --otherIndex)
          {
            CellFaceIdPacker::Unpack(
              cellAndFaceIdOfFacesInHash[otherIndex], otherCellId, otherFaceId);
            viskores::exec::CellFaceCanonicalId(otherFaceId, cellSet.GetCellShape(otherCellId),
              cellSet.GetIndices(otherCellId), otherFace);
            if (myFace[0] == otherFace[0] && myFace[1] == otherFace[1] && myFace[2] == otherFace[2])
            {
              // Faces are the same. Must be internal. We don't have to worry about otherFace
              // matching anything else because a proper topology will have at most 2 cells sharing
              // a face, so there should be no more matches.
              isInternal = true;
              break;
            }
          }
          if (isInternal) // If two faces are internal,
          {               // swap them to the end of the list to avoid revisiting them.
            --myIndex;    // decrement for the first duplicate face, which is at the end
            if (myIndex != otherIndex)
            {
              FaceCounts::SwapFace<CellFaceIdPacker::CellAndFaceIdType>(
                cellAndFaceIdOfFacesInHash[otherIndex], cellAndFaceIdOfFacesInHash[myIndex]);
            }
            --myIndex; // decrement for the second duplicate face
          }
          else // If the face is external
          {    // swap it to the front of the list, to avoid revisiting it.
            if (myIndex != numExternalFaces)
            {
              FaceCounts::SwapFace<CellFaceIdPacker::CellAndFaceIdType>(
                cellAndFaceIdOfFacesInHash[myIndex], cellAndFaceIdOfFacesInHash[numExternalFaces]);
            }
            ++numExternalFaces; // increment for the new external face
            // myIndex remains the same, since we have a new face to check at the same myIndex.
            // However, numExternalFaces has incremented, so the loop could still terminate.
          }
        }
        return numExternalFaces;
      }
    }

  private:
    template <typename FaceT, typename FaceRefT>
    VISKORES_EXEC inline static void SwapFace(FaceRefT&& cellAndFace1, FaceRefT&& cellAndFace2)
    {
      const FaceT tmpCellAndFace = cellAndFace1;
      cellAndFace1 = cellAndFace2;
      cellAndFace2 = tmpCellAndFace;
    }
  };

public:
  // Worklet that returns the number of points for each outputted face.
  // Have to manage the case where multiple faces have the same hash.
  class NumPointsPerFace : public viskores::worklet::WorkletMapField
  {
  public:
    using ControlSignature = void(FieldIn cellAndFaceIdOfFacesInHash, WholeCellSetIn<> inputCells,
      FieldOut numPointsInExternalFace);
    using ExecutionSignature = void(_1, _2, VisitIndex, _3);
    using InputDomain = _1;

    using ScatterType = viskores::worklet::ScatterCounting;

    template <typename CellAndFaceIdOfFacesInHash, typename CellSetType>
    VISKORES_EXEC void operator()(const CellAndFaceIdOfFacesInHash& cellAndFaceIdOfFacesInHash,
      const CellSetType& cellSet, viskores::IdComponent visitIndex,
      viskores::IdComponent& numPointsInExternalFace) const
    {
      // external faces are first, so we can use the visit index directly
      CellFaceIdPacker::CellIdType myCellId;
      CellFaceIdPacker::FaceIdType myFaceId;
      CellFaceIdPacker::Unpack(cellAndFaceIdOfFacesInHash[visitIndex], myCellId, myFaceId);

      viskores::exec::CellFaceNumberOfPoints(
        myFaceId, cellSet.GetCellShape(myCellId), numPointsInExternalFace);
    }
  };

  // Worklet that returns the shape and connectivity for each external face
  class BuildConnectivity : public viskores::worklet::WorkletMapField
  {
  public:
    using ControlSignature = void(FieldIn cellAndFaceIdOfFacesInHash, WholeCellSetIn<> inputCells,
      FieldOut shapesOut, FieldOut connectivityOut, FieldOut cellIdMapOut);
    using ExecutionSignature = void(_1, _2, VisitIndex, _3, _4, _5);
    using InputDomain = _1;

    using ScatterType = viskores::worklet::ScatterCounting;

    template <typename CellAndFaceIdOfFacesInHash, typename CellSetType, typename ConnectivityType>
    VISKORES_EXEC void operator()(const CellAndFaceIdOfFacesInHash& cellAndFaceIdOfFacesInHash,
      const CellSetType& cellSet, viskores::IdComponent visitIndex, viskores::UInt8& shapeOut,
      ConnectivityType& connectivityOut, viskores::Id& cellIdMapOut) const
    {
      // external faces are first, so we can use the visit index directly
      CellFaceIdPacker::CellIdType myCellId;
      CellFaceIdPacker::FaceIdType myFaceId;
      CellFaceIdPacker::Unpack(cellAndFaceIdOfFacesInHash[visitIndex], myCellId, myFaceId);

      const typename CellSetType::CellShapeTag shapeIn = cellSet.GetCellShape(myCellId);
      viskores::exec::CellFaceShape(myFaceId, shapeIn, shapeOut);
      cellIdMapOut = myCellId;

      viskores::IdComponent numFacePoints;
      viskores::exec::CellFaceNumberOfPoints(myFaceId, shapeIn, numFacePoints);
      VISKORES_ASSERT(numFacePoints == connectivityOut.GetNumberOfComponents());

      const typename CellSetType::IndicesType inCellIndices = cellSet.GetIndices(myCellId);
      for (viskores::IdComponent facePointIndex = 0; facePointIndex < numFacePoints;
        ++facePointIndex)
      {
        viskores::IdComponent localFaceIndex;
        const viskores::ErrorCode status =
          viskores::exec::CellFaceLocalIndex(facePointIndex, myFaceId, shapeIn, localFaceIndex);
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
  ExternalFacesHashCountFnv1a() {}

  void ReleaseCellMapArrays() { this->CellIdMap.ReleaseResources(); }

  ///////////////////////////////////////////////////
  /// \brief ExternalFacesHashCountFnv1a: Extract Faces on outside of geometry
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

    // create an invoker
    viskores::cont::Invoker invoke;

    // Create an array to store the number of faces per cell
    viskores::cont::ArrayHandle<viskores::IdComponent> numFacesPerCell;

    // Compute the number of faces per cell
    viskores::cont::Timer timer;
    timer.Start();
    invoke(NumFacesPerCell(), inCellSet, numFacesPerCell);
    timer.Stop();
    log.AddDictionaryEntry("seconds-num-faces-per-cell", timer.GetElapsedTime());

    // Compute the offsets into a packed array holding face information for each cell.
    viskores::Id totalNumberOfFaces;
    viskores::cont::ArrayHandle<viskores::Id> facesPerCellOffsets;
    timer.Start();
    viskores::cont::ConvertNumComponentsToOffsets(
      numFacesPerCell, facesPerCellOffsets, totalNumberOfFaces);
    timer.Stop();
    log.AddDictionaryEntry("seconds-face-per-cell-count", timer.GetElapsedTime());
    // Release the resources of numFacesPerCell that is not needed anymore
    numFacesPerCell.ReleaseResources();

    if (totalNumberOfFaces == 0)
    {
      // Data has no faces. Output is empty.
      outCellSet.PrepareToAddCells(0, 0);
      outCellSet.CompleteAddingCells(inCellSet.GetNumberOfPoints());
      return;
    }

    // Create an array to store the hash values of the faces
    viskores::cont::ArrayHandle<viskores::HashType> faceHashes;
    faceHashes.Allocate(totalNumberOfFaces);

    // Create a group vec array to access the faces of each cell conveniently
    auto faceHashesGroupVec =
      viskores::cont::make_ArrayHandleGroupVecVariable(faceHashes, facesPerCellOffsets);

    // Compute the hash values of the faces
    const viskores::Id numberOfHashes = inCellSet.GetNumberOfPoints();
    timer.Start();
    invoke(FaceHash(numberOfHashes), inCellSet, faceHashesGroupVec);
    timer.Stop();
    log.AddDictionaryEntry("seconds-face-hash", timer.GetElapsedTime());

    // Create an array to store the number of faces per hash
    viskores::cont::ArrayHandle<viskores::IdComponent> numFacesPerHash;
    numFacesPerHash.AllocateAndFill(numberOfHashes, 0);

    // Count the number of faces per hash
    timer.Start();
    invoke(NumFacesPerHash(), faceHashes, numFacesPerHash);
    timer.Stop();
    log.AddDictionaryEntry("seconds-num-faces-per-hash", timer.GetElapsedTime());

    // Compute the offsets for a packed array holding face information for each hash.
    viskores::cont::ArrayHandle<viskores::Id> facesPerHashOffsets;
    timer.Start();
    viskores::cont::ConvertNumComponentsToOffsets(numFacesPerHash, facesPerHashOffsets);
    timer.Stop();
    log.AddDictionaryEntry("seconds-face-per-hash-count", timer.GetElapsedTime());

    // Create an array to store the cell and face ids of each face per hash
    viskores::cont::ArrayHandle<CellFaceIdPacker::CellAndFaceIdType> cellAndFaceIdOfFacesPerHash;
    cellAndFaceIdOfFacesPerHash.Allocate(totalNumberOfFaces);

    // Create a group vec array to access/write the cell and face ids of each face per hash
    auto cellAndFaceIdOfFacesPerHashGroupVec = viskores::cont::make_ArrayHandleGroupVecVariable(
      cellAndFaceIdOfFacesPerHash, facesPerHashOffsets);

    // Build the cell and face ids of all faces per hash
    timer.Start();
    invoke(BuildFacesPerHash(), faceHashesGroupVec, numFacesPerHash,
      cellAndFaceIdOfFacesPerHashGroupVec);
    timer.Stop();
    log.AddDictionaryEntry("seconds-build-faces-per-hash", timer.GetElapsedTime());
    // Release the resources of the arrays that are not needed anymore
    facesPerCellOffsets.ReleaseResources();
    faceHashes.ReleaseResources();
    numFacesPerHash.ReleaseResources();

    // Create an array to count the number of external faces per hash
    viskores::cont::ArrayHandle<viskores::IdComponent> numExternalFacesPerHash;
    numExternalFacesPerHash.Allocate(numberOfHashes);

    // Compute the number of external faces per hash
    timer.Start();
    invoke(FaceCounts(), cellAndFaceIdOfFacesPerHashGroupVec, inCellSet, numExternalFacesPerHash);
    timer.Stop();
    log.AddDictionaryEntry("seconds-face-counts", timer.GetElapsedTime());

    // Create a scatter counting object to only access the hashes with external faces
    timer.Start();
    viskores::worklet::ScatterCounting scatterCullInternalFaces(numExternalFacesPerHash);
    timer.Stop();
    log.AddDictionaryEntry("seconds-scatter-cull-internal-faces", timer.GetElapsedTime());
    const viskores::Id numberOfExternalFaces =
      scatterCullInternalFaces.GetOutputRange(numberOfHashes);
    // Release the resources of externalFacesPerHash that is not needed anymore
    numExternalFacesPerHash.ReleaseResources();

    // Create an array to store the number of points of the external faces
    PointCountArrayType numPointsPerExternalFace;
    numPointsPerExternalFace.Allocate(numberOfExternalFaces);

    // Compute the number of points of the external faces
    timer.Start();
    invoke(NumPointsPerFace(), scatterCullInternalFaces, cellAndFaceIdOfFacesPerHashGroupVec,
      inCellSet, numPointsPerExternalFace);
    timer.Stop();
    log.AddDictionaryEntry("seconds-points-per-face", timer.GetElapsedTime());

    // Compute the offsets for a packed array holding the point connections for each external
    // face.
    OffsetsArrayType pointsPerExternalFaceOffsets;
    viskores::Id connectivitySize;
    timer.Start();
    viskores::cont::ConvertNumComponentsToOffsets(
      numPointsPerExternalFace, pointsPerExternalFaceOffsets, connectivitySize);
    timer.Stop();
    log.AddDictionaryEntry("seconds-face-point-count", timer.GetElapsedTime());

    // Create an array to connectivity of the external faces
    ConnectivityArrayType externalFacesConnectivity;
    externalFacesConnectivity.Allocate(connectivitySize);

    // Create a group vec array to access the connectivity of each external face
    auto externalFacesConnectivityGroupVec = viskores::cont::make_ArrayHandleGroupVecVariable(
      externalFacesConnectivity, pointsPerExternalFaceOffsets);

    // Create an array to store the shape of the external faces
    ShapeArrayType externalFacesShapes;
    externalFacesShapes.Allocate(numberOfExternalFaces);

    // Create an array to store the cell id of the external faces
    viskores::cont::ArrayHandle<viskores::Id> faceToCellIdMap;
    faceToCellIdMap.Allocate(numberOfExternalFaces);

    // Build the connectivity of the external faces
    timer.Start();
    invoke(BuildConnectivity(), scatterCullInternalFaces, cellAndFaceIdOfFacesPerHashGroupVec,
      inCellSet, externalFacesShapes, externalFacesConnectivityGroupVec, faceToCellIdMap);
    timer.Stop();
    log.AddDictionaryEntry("seconds-build-connectivity", timer.GetElapsedTime());

    outCellSet.Fill(inCellSet.GetNumberOfPoints(), externalFacesShapes, externalFacesConnectivity,
      pointsPerExternalFaceOffsets);
    this->CellIdMap = faceToCellIdMap;
  }

  viskores::cont::ArrayHandle<viskores::Id> GetCellIdMap() const { return this->CellIdMap; }

private:
  viskores::cont::ArrayHandle<viskores::Id> CellIdMap;

}; // struct ExternalFacesHashCountFnv1a
};
} // namespace viskores::worklet

#endif // viskores_worklet_ExternalFacesHashCountFnv1a_h
