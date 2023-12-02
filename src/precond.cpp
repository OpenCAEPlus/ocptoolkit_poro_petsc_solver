#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "PETScSolver.h"

using namespace std;

PetscErrorCode precondApplyMSP(PC pc, Vec xin, Vec xout)
{

    FILE *fp;
    fp = fopen("linear_solver.log", "a");
    // Get A from PC
    shellContext *context;
    PCShellGetContext(pc, (void **)&context);
    Mat Atmp = (*context).BMat;
    Mat App = (*context).App;
    Mat Ass = (*context).Ass;

    PS_SLL *allLower = (*context).lower;
    PS_SLL *allUpper = (*context).upper;

    int myid;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    PS_SLL iStart = allLower[myid];
    PS_SLL iEnd = allUpper[myid];
    PS_SLL local_size = iEnd - iStart + 1;

    PS_SLL nb = (*context).nb;

    double start, finish;

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    // Solve App*Xp = Bp
    Vec Xp;
    Vec Bp;
    get_Prhs(xin, Bp, local_size, iStart, nb);
    get_Prhs(xout, Xp, local_size, iStart, nb);

    preSolver(App, Bp, Xp, true, PRE_AMG);

    finish = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    if (myid == 0)
        fprintf(fp, "App presolver time: %lf Sec\n", finish - start);

    start = MPI_Wtime();
    // Solve Ass*Xs = Bs
    Vec Xs;
    Vec Bs;
    get_Srhs(xin, Bs, local_size, iStart, nb);
    get_Srhs(xout, Xs, local_size, iStart, nb);

    // preSolver( Ass, Bs, Xs, true, PRE_BJACOBI);
    preSolver(Ass, Bs, Xs, true, PRE_BJACOBI);

    finish = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    if (myid == 0)
        fprintf(fp, "Ass PRE_BJACOBI time: %lf Sec\n", finish - start);

    start = MPI_Wtime();
    // Combine Psol and Ssol to sol
    combine_PS(Xp, Xs, xout, local_size, iStart, nb);

    // Smooth xout
    // preSolver( Atmp, xin, xout, false, PRE_BJACOBI);
    preSolver(Atmp, xin, xout, false, PRE_BJACOBI);

    finish = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    if (myid == 0)
        fprintf(fp, "Atmp PRE_BJACOBI time: %lf Sec\n", finish - start);

    PetscErrorCode ierr = VecDestroy(&Bp);
    CHKERRQ(ierr);
    ierr = VecDestroy(&Xp);
    CHKERRQ(ierr);
    ierr = VecDestroy(&Bs);
    CHKERRQ(ierr);
    ierr = VecDestroy(&Xs);
    CHKERRQ(ierr);

    fclose(fp);

    return 0;
}

PetscErrorCode precondApplyCPR(PC pc, Vec xin, Vec xout)
{

    FILE *fp;
    fp = fopen("linear_solver.log", "a");
    // Get A from PC
    shellContext *context;
    PCShellGetContext(pc, (void **)&context);
    Mat Atmp = (*context).BMat;
    Mat App = (*context).App;
    // Mat Ass = (*context).Ass;

    PS_SLL *allLower = (*context).lower;
    PS_SLL *allUpper = (*context).upper;

    int myid;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    PS_SLL iStart = allLower[myid];
    PS_SLL iEnd = allUpper[myid];
    PS_SLL local_size = iEnd - iStart + 1;

    PS_SLL nb = (*context).nb;

    double start, finish;

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    // Solve App*Xp = Bp
    Vec Xp;
    Vec Bp;
    get_Prhs(xin, Bp, local_size, iStart, nb);
    get_Prhs(xout, Xp, local_size, iStart, nb);

    preSolver(App, Bp, Xp, true, PRE_AMG);

    finish = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    if (myid == 0)
        fprintf(fp, "App PRE_AMG time: %lf Sec\n", finish - start);

    // start = MPI_Wtime();
    // Solve Ass*Xs = Bs
    // Vec Xs;
    // Vec Bs;
    // get_Srhs(xin, Bs, local_size, iStart, nb);
    // get_Srhs(xout, Xs, local_size, iStart, nb);

    // // preSolver( Ass, Bs, Xs, true, PRE_BJACOBI);
    // preSolver(Ass, Bs, Xs, true, PRE_BJACOBI);

    // finish = MPI_Wtime();
    // MPI_Barrier(MPI_COMM_WORLD);
    // if (myid == 0)
    //     fprintf(fp, "Ass PRE_BJACOBI time: %lf Sec\n", finish - start);

    start = MPI_Wtime();
    // prolongation Psol to sol
    combine_P(Xp, xout, local_size, iStart, nb);

    // Smooth xout
    // preSolver( Atmp, xin, xout, false, PRE_BJACOBI);
    preSolver(Atmp, xin, xout, false, PRE_BJACOBI); // PRE_PILUT, PRE_ILU, PRE_EU

    finish = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    if (myid == 0)
        fprintf(fp, "Atmp PRE_BJACOBI time: %lf Sec\n", finish - start);

    PetscErrorCode ierr = VecDestroy(&Bp);
    CHKERRQ(ierr);
    ierr = VecDestroy(&Xp);
    CHKERRQ(ierr);
    // ierr = VecDestroy(&Bs);
    // CHKERRQ(ierr);
    // ierr = VecDestroy(&Xs);
    // CHKERRQ(ierr);

    fclose(fp);

    return 0;
}

// ********************************************
// parameter:
//     myid: 当前进程号
//     num_procs: 进程总数
//     nrow、nb、rpt: BSR格式矩阵
//     allLower: 每一个分段的起始位置
//     allUpper: 每一个分段的终止位置
//     allDisp: 每一个分段的偏移量
//     allSize: 每一个分段的分段长度
// ********************************************

void calBLowerUpper(int myid, PS_SLL num_procs, PS_SLL nrow, PS_SLL nb, PS_SLL *rpt, PS_SLL &local_size, PS_SLL *allLower, PS_SLL *allUpper, PS_SLL *allDisp, PS_SLL *allSize)
{
    PS_SLL i;

    // 计算每个进程上分到的块矩阵的行数
    local_size = nrow / num_procs;

    // 因无法整除而余下的行数
    PS_SLL tmpExtra = nrow - local_size * num_procs;

    // 定义数组，用来表示每一个分段的起始位置、终止位置、偏移量、分段长度
    for (i = 0; i < num_procs; i++)
    {
        allLower[i] = local_size * i;
        allLower[i] += PS_MIN(i, tmpExtra);

        allUpper[i] = local_size * (i + 1);
        allUpper[i] += PS_MIN(i + 1, tmpExtra) - 1;

        allDisp[i] = rpt[allLower[i]];
        allSize[i] = rpt[allUpper[i] + 1] - rpt[allLower[i]];
    }

    local_size = allUpper[myid] - allLower[myid] + 1;
}

PS_SLL combine_P(Vec Psol, Vec sol, PS_SLL nBlockRows, PS_SLL iStart, PS_SLL nb)
{
    // VecRestoreSubVector
    PS_SLL *pIndex = (PS_SLL *)malloc(nBlockRows * sizeof(PS_SLL));
    for (PS_SLL i = 0; i < nBlockRows; i++)
    {
        pIndex[i] = i + iStart;
    }
    double *Pvec = (double *)malloc(nBlockRows * sizeof(double));
    VecGetValues(Psol, nBlockRows, pIndex, Pvec);

    for (PS_SLL i = 0; i < nBlockRows; i++)
    {
        pIndex[i] = (i + iStart) * nb;
    }
    VecSetValues(sol, nBlockRows, pIndex, Pvec, INSERT_VALUES);

    PetscErrorCode ierr = VecAssemblyBegin(sol);
    CHKERRQ(ierr);
    ierr = VecAssemblyEnd(sol);
    CHKERRQ(ierr);

    free(pIndex);
    free(Pvec);
    return 0;
}

PS_SLL combine_PS(Vec Psol, Vec Ssol, Vec sol, PS_SLL nBlockRows, PS_SLL iStart, PS_SLL nb)
{
    // VecRestoreSubVector
    PS_SLL *pIndex = (PS_SLL *)malloc(nBlockRows * sizeof(PS_SLL));
    for (PS_SLL i = 0; i < nBlockRows; i++)
    {
        pIndex[i] = i + iStart;
    }
    double *Pvec = (double *)malloc(nBlockRows * sizeof(double));
    VecGetValues(Psol, nBlockRows, pIndex, Pvec);

    for (PS_SLL i = 0; i < nBlockRows; i++)
    {
        pIndex[i] = (i + iStart) * nb;
    }
    VecSetValues(sol, nBlockRows, pIndex, Pvec, INSERT_VALUES);

    PetscErrorCode ierr = VecAssemblyBegin(sol);
    CHKERRQ(ierr);
    ierr = VecAssemblyEnd(sol);
    CHKERRQ(ierr);

    //-------------------------------------------------------------------
    PS_SLL numS = nBlockRows * (nb - 1);
    PS_SLL *sIndex = (PS_SLL *)malloc(numS * sizeof(PS_SLL));
    for (PS_SLL i = 0; i < numS; i++)
    {
        sIndex[i] = i + iStart * (nb - 1);
    }
    double *Svec = (double *)malloc(numS * sizeof(double));
    VecGetValues(Ssol, numS, sIndex, Svec);

    PS_SLL index = 0;

    for (PS_SLL i = 0; i < nBlockRows; i++)
    {
        for (PS_SLL j = 1; j < nb; j++)
        {
            sIndex[index] = (iStart + i) * nb + j;
            index++;
        }
    }

    VecSetValues(sol, numS, sIndex, Svec, INSERT_VALUES);

    ierr = VecAssemblyBegin(sol);
    CHKERRQ(ierr);
    ierr = VecAssemblyEnd(sol);
    CHKERRQ(ierr);

    free(pIndex);
    free(Pvec);
    free(sIndex);
    free(Svec);
    return 0;
}

PS_SLL get_Prhs(Vec globalVec, Vec &localVec, PS_SLL nBlockRows, PS_SLL iStart, PS_SLL nb)
{
    IS is;
    PetscErrorCode ierr = ISCreateStride(PETSC_COMM_WORLD, nBlockRows, iStart * nb, nb, &is);
    CHKERRQ(ierr);
    ierr = VecGetSubVector(globalVec, is, &localVec);
    CHKERRQ(ierr);

    ISDestroy(&is);
    return 0;
}

PS_SLL get_Srhs(Vec globalVec, Vec &localVec, PS_SLL nBlockRows, PS_SLL iStart, PS_SLL nb)
{
    PS_SLL *idx = (PS_SLL *)malloc(sizeof(PS_SLL) * nBlockRows * (nb - 1));
    PS_SLL index = 0;
    for (PS_SLL i = 0; i < nBlockRows; i++)
    {
        for (PS_SLL j = 1; j < nb; j++)
        {
            idx[index] = iStart + i * nb + j;
            index++;
        }
    }
    IS is;
    PetscErrorCode ierr = ISCreateGeneral(PETSC_COMM_WORLD, nBlockRows * (nb - 1), idx, PETSC_COPY_VALUES, &is);
    CHKERRQ(ierr);
    ierr = VecGetSubVector(globalVec, is, &localVec);
    CHKERRQ(ierr);

    ISDestroy(&is);
    free(idx);
    return 0;
}

PS_SLL get_PP(dBSRmat_ *A, PS_SLL Istart, PS_SLL Iend, PS_SLL matrixDim, Mat &localApp)
{
    FILE *fp;
    fp = fopen("linear_solver.log", "a");
    int myid;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    double start, finish;

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    dCSRmat_ App;

    const PS_SLL row = A->ROW;
    const PS_SLL col = A->COL;
    const PS_SLL nnz = A->NNZ;
    const PS_SLL nb = A->nb;
    const PS_SLL nb2 = nb * nb;
    double *val = A->val;
    PS_SLL *IA = A->IA;
    PS_SLL *JA = A->JA;

    App.row = row;
    App.col = col;
    App.nnz = nnz;

    App.IA = (PS_SLL *)malloc(sizeof(PS_SLL) * (row + 1));
    App.JA = (PS_SLL *)malloc(sizeof(PS_SLL) * nnz);
    App.val = (double *)malloc(sizeof(double) * nnz);

    double *Pval = App.val;

    memcpy(App.IA, IA, (row + 1) * sizeof(PS_SLL));
    memcpy(App.JA, JA, nnz * sizeof(PS_SLL));

    PS_SLL i, j, Ii;
    for (i = 0; i < nnz; ++i)
    {
        Pval[i] = val[i * nb2];
    }

    finish = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    if (myid == 0)
        fprintf(fp, "get_PP_fasp time: %lf Sec\n", finish - start);
    //-------------------------------------------

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    Mat matApp;

    PS_SLL blockSize = 1;
    PS_SLL rowWidth = App.row * blockSize;
    PS_SLL nBlockRows = App.row;

    PS_SLL *rpt = App.IA;
    PS_SLL *cpt = App.JA;
    val = App.val;

    PS_SLL *nDCount = (PS_SLL *)malloc(sizeof(PS_SLL) * nBlockRows);
    PS_SLL *nNDCount = (PS_SLL *)malloc(sizeof(PS_SLL) * nBlockRows);

    for (i = 0; i < nBlockRows; i++)
    {
        nDCount[i] = 0;
        for (j = rpt[i]; j < rpt[i + 1]; j++)
        {
            if (cpt[j] >= Istart && cpt[j] <= Iend)
            {
                nDCount[i]++;
            }
        }
    }

    for (i = 0; i < nBlockRows; i++)
    {
        nNDCount[i] = rpt[i + 1] - rpt[i] - nDCount[i];
    }

    PS_SLL dim = matrixDim / nb;
    PetscErrorCode ierr = MatCreateBAIJ(PETSC_COMM_WORLD, blockSize, rowWidth, rowWidth, dim, dim, 0, nDCount, 0, nNDCount, &matApp);
    CHKERRQ(ierr);

    ierr = MatSetFromOptions(matApp);
    CHKERRQ(ierr);

    ierr = MatSetUp(matApp);
    CHKERRQ(ierr);

    double *valpt = val;
    PS_SLL *globalx = (PS_SLL *)malloc(blockSize * sizeof(PS_SLL));
    PS_SLL *globaly = (PS_SLL *)malloc(blockSize * sizeof(PS_SLL));
    PS_SLL b2 = blockSize * blockSize;

    for (Ii = 0; Ii < nBlockRows; Ii++)
    {
        for (i = 0; i < blockSize; i++)
        {
            globalx[i] = (Ii + Istart) * blockSize + i;
        }

        for (i = rpt[Ii]; i < rpt[Ii + 1]; i++)
        {

            for (j = 0; j < blockSize; j++)
            {
                globaly[j] = cpt[i] * blockSize + j;
            }
            ierr = MatSetValues(matApp, blockSize, globalx, blockSize, globaly, valpt, INSERT_VALUES);
            CHKERRQ(ierr);
            valpt += b2;
        }
    }

    ierr = MatAssemblyBegin(matApp, MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);
    ierr = MatAssemblyEnd(matApp, MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);

    localApp = matApp;

    finish = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    if (myid == 0)
        fprintf(fp, "get_PP_petsc time: %lf Sec\n", finish - start);

    fclose(fp);

    free(App.IA);
    free(App.JA);
    free(App.val);
    return 0;
}

PS_SLL get_SS(dBSRmat_ *A, PS_SLL Istart, PS_SLL Iend, PS_SLL matrixDim, Mat &localAss)
{
    FILE *fp;
    fp = fopen("linear_solver.log", "a");
    int myid;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    double start, finish;
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    dBSRmat_ Ass;

    const PS_SLL ROW = A->ROW;
    const PS_SLL COL = A->COL;
    const PS_SLL NNZ = A->NNZ;
    const PS_SLL nb = A->nb;
    const PS_SLL nb2 = nb * nb;
    double *val = A->val;
    PS_SLL *IA = A->IA;
    PS_SLL *JA = A->JA;

    PS_SLL nbs = nb - 1;
    PS_SLL nbs2 = nbs * nbs;

    Ass.ROW = ROW;
    Ass.COL = COL;
    Ass.NNZ = NNZ;
    Ass.nb = nbs;

    Ass.IA = (PS_SLL *)malloc(sizeof(PS_SLL) * (ROW + 1));
    Ass.JA = (PS_SLL *)malloc(sizeof(PS_SLL) * NNZ);
    Ass.val = (double *)malloc(sizeof(double) * nbs2 * NNZ);
    double *Sval = Ass.val;

    memcpy(Ass.IA, IA, (ROW + 1) * sizeof(PS_SLL));
    memcpy(Ass.JA, JA, NNZ * sizeof(PS_SLL));

    PS_SLL i, j, k, Ii;
    for (i = 0; i < NNZ; ++i)
    {
        for (j = 0; j < nbs; ++j)
        {
            for (k = 0; k < nbs; ++k)
            {
                Sval[i * nbs2 + j * nbs + k] = val[i * nb2 + (j + 1) * nb + k + 1];
            }
        }
    }
    finish = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    if (myid == 0)
        fprintf(fp, "get_SS_fasp time: %lf Sec\n", finish - start);

    //-------------------------------------------
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    Mat matAss;

    PS_SLL blockSize = Ass.nb;
    PS_SLL rowWidth = Ass.ROW * blockSize;
    PS_SLL nBlockRows = Ass.ROW;

    PS_SLL *rpt = Ass.IA;
    PS_SLL *cpt = Ass.JA;
    val = Ass.val;

    PS_SLL *nDCount = (PS_SLL *)malloc(sizeof(PS_SLL) * nBlockRows);
    PS_SLL *nNDCount = (PS_SLL *)malloc(sizeof(PS_SLL) * nBlockRows);

    for (i = 0; i < nBlockRows; i++)
    {
        nDCount[i] = 0;
        for (j = rpt[i]; j < rpt[i + 1]; j++)
        {
            if (cpt[j] >= Istart && cpt[j] <= Iend)
            {
                nDCount[i]++;
            }
        }
    }

    for (i = 0; i < nBlockRows; i++)
    {
        nNDCount[i] = rpt[i + 1] - rpt[i] - nDCount[i];
    }

    PS_SLL dim = matrixDim / nb * (nb - 1);
    PetscErrorCode ierr = MatCreateBAIJ(PETSC_COMM_WORLD, blockSize, rowWidth, rowWidth, dim, dim, 0, nDCount, 0, nNDCount, &matAss);
    CHKERRQ(ierr);

    ierr = MatSetFromOptions(matAss);
    CHKERRQ(ierr);

    ierr = MatSetUp(matAss);
    CHKERRQ(ierr);

    double *valpt = val;
    PS_SLL *globalx = (PS_SLL *)malloc(blockSize * sizeof(PS_SLL));
    PS_SLL *globaly = (PS_SLL *)malloc(blockSize * sizeof(PS_SLL));
    PS_SLL b2 = blockSize * blockSize;

    for (Ii = 0; Ii < nBlockRows; Ii++)
    {
        for (i = 0; i < blockSize; i++)
        {
            globalx[i] = (Ii + Istart) * blockSize + i;
        }

        for (i = rpt[Ii]; i < rpt[Ii + 1]; i++)
        {

            for (j = 0; j < blockSize; j++)
            {
                globaly[j] = cpt[i] * blockSize + j;
            }
            ierr = MatSetValues(matAss, blockSize, globalx, blockSize, globaly, valpt, INSERT_VALUES);
            CHKERRQ(ierr);
            valpt += b2;
        }
    }

    ierr = MatAssemblyBegin(matAss, MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);
    ierr = MatAssemblyEnd(matAss, MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);

    localAss = matAss;

    finish = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    if (myid == 0)
        fprintf(fp, "get_SS_petsc time: %lf Sec\n", finish - start);
    fclose(fp);

    free(Ass.IA);
    free(Ass.JA);
    free(Ass.val);

    return 0;
}

/**
 * A =
 * [PP  PN1  PN2  ... PT ]
 * [N1P N1N1 N1N2 ... N1T]
 * [N2P N2N1 N2N2 ... N2T]
 *         ...
 * [TP  TN1  TN2  ... TT]
 *
 *
 * */
// PS_SLL get_SS_thermal(dBSRmat_ *A, PS_SLL Istart, PS_SLL Iend, PS_SLL matrixDim, Mat &localAss)
// {
//     FILE *fp;
//     fp = fopen("linear_solver.log", "a");
//     int myid;
//     MPI_Comm_rank(MPI_COMM_WORLD, &myid);
//     double start, finish;
//     MPI_Barrier(MPI_COMM_WORLD);
//     start = MPI_Wtime();

//     dBSRmat_ Ass;

//     const PS_SLL ROW = A->ROW;
//     const PS_SLL COL = A->COL;
//     const PS_SLL NNZ = A->NNZ;
//     const PS_SLL nb = A->nb;
//     const PS_SLL nb2 = nb * nb;
//     double *val = A->val;
//     PS_SLL *IA = A->IA;
//     PS_SLL *JA = A->JA;

//     PS_SLL nbs = nb - 1;
//     PS_SLL nbs2 = nbs * nbs;

//     Ass.ROW = ROW;
//     Ass.COL = COL;
//     Ass.NNZ = NNZ;
//     Ass.nb = nbs;

//     Ass.IA = (PS_SLL *)malloc(sizeof(PS_SLL) * (ROW + 1));
//     Ass.JA = (PS_SLL *)malloc(sizeof(PS_SLL) * NNZ);
//     Ass.val = (double *)malloc(sizeof(double) * nbs2 * NNZ);
//     double *Sval = Ass.val;

//     memcpy(Ass.IA, IA, (ROW + 1) * sizeof(PS_SLL));
//     memcpy(Ass.JA, JA, NNZ * sizeof(PS_SLL));

//     PS_SLL i, j, k, Ii;
//     for (i = 0; i < NNZ; ++i)
//     {
//         for (j = 0; j < nbs; ++j)
//         {
//             for (k = 0; k < nbs; ++k)
//             {
//                 Sval[i * nbs2 + j * nbs + k] = val[i * nb2 + (j + 1) * nb + k + 1];
//             }
//         }
//     }
//     finish = MPI_Wtime();
//     MPI_Barrier(MPI_COMM_WORLD);
//     if (myid == 0)
//         fprintf(fp, "get_SS_fasp time: %lf Sec\n", finish - start);

//     //-------------------------------------------
//     MPI_Barrier(MPI_COMM_WORLD);
//     start = MPI_Wtime();

//     Mat matAss;

//     PS_SLL blockSize = Ass.nb;
//     PS_SLL rowWidth = Ass.ROW * blockSize;
//     PS_SLL nBlockRows = Ass.ROW;

//     PS_SLL *rpt = Ass.IA;
//     PS_SLL *cpt = Ass.JA;
//     val = Ass.val;

//     PS_SLL *nDCount = (PS_SLL *)malloc(sizeof(PS_SLL) * nBlockRows);
//     PS_SLL *nNDCount = (PS_SLL *)malloc(sizeof(PS_SLL) * nBlockRows);

//     for (i = 0; i < nBlockRows; i++)
//     {
//         nDCount[i] = 0;
//         for (j = rpt[i]; j < rpt[i + 1]; j++)
//         {
//             if (cpt[j] >= Istart && cpt[j] <= Iend)
//             {
//                 nDCount[i]++;
//             }
//         }
//     }

//     for (i = 0; i < nBlockRows; i++)
//     {
//         nNDCount[i] = rpt[i + 1] - rpt[i] - nDCount[i];
//     }

//     PS_SLL dim = matrixDim / nb * (nb - 1);
//     PetscErrorCode ierr = MatCreateBAIJ(PETSC_COMM_WORLD, blockSize, rowWidth, rowWidth, dim, dim, 0, nDCount, 0, nNDCount, &matAss);
//     CHKERRQ(ierr);

//     ierr = MatSetFromOptions(matAss);
//     CHKERRQ(ierr);

//     ierr = MatSetUp(matAss);
//     CHKERRQ(ierr);

//     double *valpt = val;
//     PS_SLL *globalx = (PS_SLL *)malloc(blockSize * sizeof(PS_SLL));
//     PS_SLL *globaly = (PS_SLL *)malloc(blockSize * sizeof(PS_SLL));
//     PS_SLL b2 = blockSize * blockSize;

//     for (Ii = 0; Ii < nBlockRows; Ii++)
//     {
//         for (i = 0; i < blockSize; i++)
//         {
//             globalx[i] = (Ii + Istart) * blockSize + i;
//         }

//         for (i = rpt[Ii]; i < rpt[Ii + 1]; i++)
//         {

//             for (j = 0; j < blockSize; j++)
//             {
//                 globaly[j] = cpt[i] * blockSize + j;
//             }
//             ierr = MatSetValues(matAss, blockSize, globalx, blockSize, globaly, valpt, INSERT_VALUES);
//             CHKERRQ(ierr);
//             valpt += b2;
//         }
//     }

//     ierr = MatAssemblyBegin(matAss, MAT_FINAL_ASSEMBLY);
//     CHKERRQ(ierr);
//     ierr = MatAssemblyEnd(matAss, MAT_FINAL_ASSEMBLY);
//     CHKERRQ(ierr);

//     localAss = matAss;

//     finish = MPI_Wtime();
//     MPI_Barrier(MPI_COMM_WORLD);
//     if (myid == 0)
//         fprintf(fp, "get_SS_petsc time: %lf Sec\n", finish - start);
//     fclose(fp);

//     free(Ass.IA);
//     free(Ass.JA);
//     free(Ass.val);

//     return 0;
// }
