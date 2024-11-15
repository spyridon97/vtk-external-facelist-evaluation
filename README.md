# vtk-external-facelist-evaluation

## Introduction

Repository for the evaluation of the external facelist calculation algorithms in VTK/VTk-m.

The algorithms that are evaluated are the following:

1. VTK's S-Classifier found in src/vtkGeometryFilterSClassifier
2. VTK's S-Hash found in src/vtkDataSetSurfaceFilterSHash
3. VTK's P-Classifier found in src/vtkGeometryFilterPClassifier
4. VTK's P-Hash found in src/vtkGeometryFilterPHash
5. VTK-m's P-Hash-Sort with FNV1A found in src/ExternalFacesHashSortFnv1a
6. VTK-m's P-Hash-Sort with MinPointID found in src/ExternalFacesHashSortMinPointId
7. VTK-m's P-Hash-Fight with FNV1A found in src/ExternalFacesHashFightFnv1a
8. VTK-m's P-Hash-Fight with MinPointID found in src/ExternalFacesHashFightMinPointId
9. VTK-m's P-Hash-Count with FNV1A found in src/ExternalFacesHashCountFnv1a
10. VTK-m's P-Hash-Count with MinPointID found in src/ExternalFacesHashCountMinPointId

## Compilation

To compile the executable on the frontier supercomputer, you can use the script `compile_frontier.sh`.
If you are compiling locally, you can get inspiration from the `compile_frontier.sh` script, and remove or change
what you do or do not need depending on your system.

## Executable

These algorithms can be used through the compiled executable named `vtk-external-facelist-evaluation`, with the
following options:

```
./vtk-external-facelist-evaluation -h
External Facelist Evaluation
Usage: ./vtk-external-facelist-evaluation [OPTIONS]

Options:
  -h,--help                   Print this help message and exit
  -i,--input TEXT REQUIRED    Input file name
  -t,--threads UINT:UINT in [1 - 128]
                              Number of threads (Default: 1)
  -d,--device TEXT            Device name. Available: "Any" "Serial" "TBB" "Kokkos" . (Default: TBB).
  -n,--trials UINT            Number of trials (Default: 1)
  -r,--randomize              Randomize connections of generated topology
  -s,--seed UINT Needs: --randomize
                              Randomized seed (Default: 1234567890)
  --hash-distribution         Run the Hash Distribution algorithm
  --s-classifier              Run the S-Classifier algorithm
  --s-hash                    Run the S-Hash algorithm
  --p-classifier              Run the P-Classifier algorithm
  --p-hash                    Run the P-Hash algorithm
  --p-hash-fight              Run the P-HashFight algorithm
  --p-hash-sort               Run the P-Hash-Sort algorithm
  --p-hash-count              Run the P-Hash-Count algorithm
  -f,--hash-function INT:INT in [0 - 2]
                              Hash function, where 0 is All, 1 is FNV1A, 2 is MinPointID (Default: 0)
```

## Python Evaluation scripts

The ```vtk-external-facelist-evaluation``` executable can be used to evaluate the algorithms.
In the `evaluation` directory, you can find the following scripts:

1. `configuration.py` is used to define information regarding where data, executable(s) and
   results should be located. Be sure to check it out.
2. `run_evaluation.py` is used to run the ```vtk-external-facelist-evaluation``` executable. It will run the
   executable with the specified options and store the different kinds of results. For different segments of the
   evaluation, you can use the ``--method`` option to part of the evaluation you want to run.
3. `generate_figures.py` is used to generate the figures based on the results obtained from the evaluation. For
   different segments of the evaluation, you can use the ``--method`` option to generate figures for a specific part of
   the evaluation.

## Data

The dataset used for the evaluations of the algorithms is stored can be downloaded from the following
[Google Drive folder](https://drive.google.com/drive/folders/1ocRUgLCLuruwD5GLGS9ZjhNbA5XVHxr6?usp=sharing).

Be sure to update the `configuration.py` file with the correct path to the data.

## Results

The evaluation data is stored in different folders in the `evaluation/results` directory. The results that are generated
are:

1. memory-footprint: The memory footprint of the algorithms.
2. cpu-time: The CPU time of all algorithms with regular connections and 1 thread, of parallel algorithms with regular
   and randomized connections using max number of threads.
3. hash-performance: The face hash distributions of the hash functions, and the cache misses for each vtk-m algorithm
   with regular connections using different hash functions and max number of threads.
4. speed-up: The speed-up of the parallel algorithms with regular connections using max number of threads.
5. gpu-time: The GPU time of the vtkm algorithms with regular and randomized connections.
