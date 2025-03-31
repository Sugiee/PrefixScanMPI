// Author: Satheyaseelan Sugieethan
// Student ID: 20318670
#include <stdio.h>
#include <stdlib.h> // Required for rand() and srand()
#include <time.h>   // Required for time()
#include <math.h>   // Required for ceil()
#include "mpi.h"    // Required for parallel programming
/* -----------------------------------------------------------------------------------------------------
 * UpPhase Function
 * -----------------------------------------------------------------------------------------------------
 * @description: Aggregating results up each level from 2 to paddedArrayLength in doubles
 
 * @param: inputArrayLength - size of input array
           subArrayLength - size of local array on each process
           paddedArrayLength - size of padded array and used as max level
           rank - represents current process
           subArray - pointer pointing to local array of process
 * ---------------------------------------------------------------------------------------------------*/
void upPhase(int inputArrayLength, int subArrayLength, int paddedArrayLength, int rank, int* subArray) {
    for (int currLevel = 2; currLevel <= paddedArrayLength; currLevel *=2) {
        for (int i = 0; i < subArrayLength; ++i) { // Iterates through subArray to decide if element
                                                   // needs to receive, send or do nothing
            int idx = rank * subArrayLength + i;   // Stores global array indexing
            int tempRecv = 0;                      // Buffer for received element   

            // Check if current element needs to aggregated to same index and within Array range  
            if(((idx+1) % currLevel) == 0 && (idx < inputArrayLength)) {
                int source = (idx - (currLevel/2)) / subArrayLength; // calculate rank of other element
                
                if (rank == source) { // If both elements on same core, can aggregate 
                    subArray[i] = subArray[i] + subArray[i-(currLevel/2)];
                    continue; 
                }

                int tag = idx; // unique identifier for MPI_Recv()
                
                // Receive element from source process
                MPI_Recv(&tempRecv, 1, MPI_INT, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                       
                subArray[i] += tempRecv; // Element received, can add
            }
            // Check if current element needs to be added to diff index and within array range
            else if ((idx+1) % (currLevel/2) == 0 && idx+(currLevel/2) < inputArrayLength) {
                int dest = (idx + (currLevel/2)) / subArrayLength; // calculate rank element lies on
                
                if (rank == dest) continue; // both elements on same core, no send required

                int tag = idx + (currLevel/2); // unique identifier for MPI_Send()                     

                // Send element to dest process 
                MPI_Send(&subArray[i], 1, MPI_INT, dest, tag, MPI_COMM_WORLD);                  
            }
        }
    }
}
/* -----------------------------------------------------------------------------------------------------
 * downPhase function
 * -----------------------------------------------------------------------------------------------------
 * @description: Aggregating results down each level from paddedArrayLength/2 to 2 in halves
 
 * @param: inputArrayLength - size of input array
           subArrayLength - size of local array on each process
           paddedArrayLength - size of padded array and used as max level
           rank - represents current process
           subArray - pointer pointing to local array of process
 * ---------------------------------------------------------------------------------------------------*/
void downPhase(int inputArrayLength, int subArrayLength, int paddedArrayLength, int rank, int* subArray) { 
    for (int currLevel = paddedArrayLength/2; currLevel >= 2; currLevel /=2) {
        for (int i = 0; i < subArrayLength; ++i) {  // Iterates through subArray to decide if element
                                                    // needs to receive, send or do nothing
            int idx = rank * subArrayLength + i;    // Stores global array indexing
            int tempRecv = 0;                       // Buffer for received element   

            // Check if current element needs to aggregated to same index and within array range
            if (((idx+1) % currLevel) == currLevel/2 && idx < inputArrayLength && idx - (currLevel/2) >= 0) {
                int source = (idx - (currLevel/2)) / subArrayLength;

                if (rank == source) { // If both elements on same core, can aggregate 
                    subArray[i] = subArray[i] + subArray[i-(currLevel/2)];
                    continue; 
                }

                int tag = idx; // unique identifier for MPI_Recv()
                
                // Receive element from source process
                MPI_Recv(&tempRecv, 1, MPI_INT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                subArray[i] += tempRecv; // Element received, can add 
            }
            // Check if current element needs to be added to diff index and within array range
            else if (((idx+1) % currLevel) == 0 && idx + (currLevel/2) < inputArrayLength ) {
                int dest = (idx + (currLevel/2)) / subArrayLength; // calculate rank element lies on
                                
                if (rank == dest) continue; // both elements on same core, no send required

                int tag = idx + currLevel/2; // unique identifier for MPI_Send()

                // Send element to dest process 
                MPI_Send(&subArray[i], 1, MPI_INT, dest, 0, MPI_COMM_WORLD); 
            }
        }
    }
}
/* -----------------------------------------------------------------------------------------------------
 * Main Programme to run Prefix Scan Programme
 * -----------------------------------------------------------------------------------------------------
 * @input: inputArrayLength - size of array to generate numbers for
 * @return: outputArray - array used to store results post prefix scan
 * ---------------------------------------------------------------------------------------------------*/
int main(int argc, char *argv[]) {
    
    MPI_Init(&argc, &argv); // Initialise MPI environment

    int rank, size; // Store process rank and total number of processes
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Array variables and pointers 
    int inputArrayLength, subArrayLength, paddedArrayLength;
    int *inputArray = NULL, *subArray = NULL, *outputArray =NULL;

    // Variables used for timing
    //clock_t start, end;

    // Array population and algorithm setup on root process
    if (rank == 0) {
        // User input
        printf("Prefix Scan programme is running!\n\nPlease enter the size of the array: ");
        fflush(stdout);
        scanf("%d", &inputArrayLength);

        //start = clock();

        // Calculate 2^n => arrayLength, also represents max level 
        paddedArrayLength = 1;
        while (paddedArrayLength < inputArrayLength) {
            paddedArrayLength <<= 1;
        }

        // Allocate memory to arrays
        inputArray = (int*) calloc(paddedArrayLength, sizeof(int));
        outputArray = (int*) malloc(inputArrayLength * sizeof(int));
        
        // calculate subArrayLength to allocate memory on all processes
        subArrayLength = ceil((float)inputArrayLength/size);

        // Seed and populate array with random numbers between 0-99
        srand(time(0));
        for (int i = 0; i < inputArrayLength; ++i) {
            inputArray[i] = rand() % 100;
        }

        // Print generated sequence 
        printf("\nGenerated sequence: ");
        fflush(stdout);
        for (int i = 0 ; i < inputArrayLength; ++i) {
            printf("%d ", inputArray[i]);
        }
        printf("\n\n");
        fflush(stdout);
    }

    // Broadcast values from root process to all processes
    MPI_Bcast(&inputArrayLength, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&paddedArrayLength, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&subArrayLength, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate memory to subArray on all processes
    subArray = (int*) calloc(subArrayLength, sizeof(int));

    // Split even blocks of inputArray to subArray on all processes 
    MPI_Scatter(inputArray, subArrayLength, MPI_INT, 
                subArray, subArrayLength, MPI_INT, 0, MPI_COMM_WORLD);  
    
    upPhase(inputArrayLength, subArrayLength, paddedArrayLength, rank, subArray);

    downPhase(inputArrayLength, subArrayLength, paddedArrayLength, rank, subArray);

    // Collect updated blocks of subArray from all processes and store in outputArray on root process
    MPI_Gather(subArray, subArrayLength, MPI_INT, 
               outputArray, subArrayLength, MPI_INT, 0, MPI_COMM_WORLD);

    // print output sequence by root process
    if (rank == 0) {
        //end = clock();
        printf("Output sequence: ");
        for (int i = 0; i < inputArrayLength; ++i) {
            printf("%d ", outputArray[i]);
        }
        fflush(stdout); 

        // double duration = (double) (end - start) / CLOCKS_PER_SEC;
        // printf("\n\nTime taken: %f\n", duration);
        // fflush(stdout); 
    }

    MPI_Finalize(); // Terminate MPI environment

    return 0;
}
/* -----------------------------------------------------------------------------------------------------
 * Test Cases for Prefix Scan Programme
 * -----------------------------------------------------------------------------------------------------
 *
 * ---------------------------------------- ** Test Case 1 ** ------------------------------------------
 *
 * N is power of 2 and N < P (Number of elements less than number of processes)
 * N = 4, P = 6 
 * inputArray = [68, 99, 66, 22]
 * outputArray = [68 167 233 255]
 *
 * ---------------------------------------- ** Test Case 2 ** ------------------------------------------
 *
 * N is not power of 2 and N < P (Number of elements less than number of processes)
 * N = 5, P = 6 
 * inputArray = [12 9 57 32 34]
 * outputArray = [12 21 78 110 144]
 *
 * ---------------------------------------- ** Test Case 3 ** ------------------------------------------
 *
 * N is power of 2 and N = P (Number of elements equal to number of processes)
 * N = 8, P = 8 
 * inputArray = [37 5 79 5 79 77 56 32]
 * outputArray = [37 42 121 126 205 282 338 370] 
 *
 * ---------------------------------------- ** Test Case 4 ** ------------------------------------------
 *
 * N is not power of 2 and N = P (Number of elements equal to number of processes)
 * N = 11, P = 11 
 * inputArray = [59 0 27 10 42 51 94 85 22 17 97]
 * outputArray = [59 59 86 96 138 189 283 368 390 407 504]
 *
 * ---------------------------------------- ** Test Case 5 ** ------------------------------------------
 *
 * N is power of 2 and N > P (Number of elements greater than number of processes)
 * N = 16, P = 6 
 * inputArray = [12 94 49 77 73 65 93 54 72 87 84 47 20 78 59 31]
 * outputArray = [12 106 155 232 305 370 463 517 589 676 760 807 827 905 964 995]
 *
 * ---------------------------------------- ** Test Case 6 ** ------------------------------------------
 *
 * N is not power of 2 and N > P (Number of elements greater than number of processes)
 * N = 17, P = 3 
 * inputArray = [66 49 65 41 37 13 38 44 85 40 32 17 71 23 15 21 39] 
 * outputArray = [66 115 180 221 258 271 309 353 438 478 510 527 598 621 636 657 696] 
 * 
 * -----------------------------------------------------------------------------------------------------
 * 
 * ---------------------------------------------------------------------------------------------------*/