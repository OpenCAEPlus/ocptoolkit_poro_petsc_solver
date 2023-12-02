#ifndef PETScSolver_h
#define PETScSolver_h

// #ifdef __cplusplus
// extern "C" {
#include "petscksp.h"
// #endif

// #ifdef __cplusplus
// }
// #endif

#define PRE_AMG 1
#define PRE_JACOBI 2
#define PRE_BJACOBI 3
#define PRE_ILU 4
#define PRE_EU 5
#define PRE_PILUT 6

#define PS_MIN(a, b) (((a) < (b)) ? (a) : (b))


#ifdef PSINT64

#define PSINTTYPEWIDTH 64

#else

#define PSINTTYPEWIDTH 32

#endif



#if PSINTTYPEWIDTH == 32
typedef int                PS_SLL;  ///< Long long signed integer

#define PSMPI_SLL         MPI_INT

#elif PSINTTYPEWIDTH == 64
typedef PetscInt64          PS_SLL;  ///< Long long signed integer

#define PSMPI_SLL         MPI_LONG_LONG_INT

#endif


typedef struct dCSRmat_
{
    PS_SLL row;
    PS_SLL col;
    PS_SLL nnz;
    PS_SLL *IA;
    PS_SLL *JA;
    double *val;
} dCSRmat_;

typedef struct dBSRmat_
{
    PS_SLL ROW;
    PS_SLL COL;
    PS_SLL NNZ;
    PS_SLL nb;
    PS_SLL storage_manner; // 0: row-major order, 1: column-major order
    double *val;
    PS_SLL *IA;
    PS_SLL *JA;
} dBSRmat_;

typedef struct dvector_
{
    PS_SLL row;
    double *val;
} dvector_;

typedef struct dBlockDiag
{
    PS_SLL ROW;
    PS_SLL nb;
    double *val;
} dBlockDiag;

typedef struct shellContext
{
    Mat BMat;
    Mat App;
    Mat Ass;
    PS_SLL nb;
    PS_SLL *lower;
    PS_SLL *upper;
} shellContext;

void calBLowerUpper(PS_SLL myid, PS_SLL num_procs, PS_SLL nrow, PS_SLL nb, PS_SLL *rpt, PS_SLL &local_size, PS_SLL *allLower, PS_SLL *allUpper, PS_SLL *allDisp, PS_SLL *allSize);

PS_SLL get_Prhs(Vec globalVec, Vec &localVec, PS_SLL nBlockRows, PS_SLL iStart, PS_SLL nb);
PS_SLL get_Srhs(Vec globalVec, Vec &localVec, PS_SLL nBlockRows, PS_SLL iStart, PS_SLL nb);

PS_SLL get_PP(dBSRmat_ *A, PS_SLL Istart, PS_SLL Iend, PS_SLL matrixDim, Mat &localApp);
PS_SLL get_SS(dBSRmat_ *A, PS_SLL Istart, PS_SLL Iend, PS_SLL matrixDim, Mat &localAss);
PS_SLL combine_PS(Vec Psol, Vec Ssol, Vec sol, PS_SLL nBlockRows, PS_SLL iStart, PS_SLL nb);
PS_SLL combine_P(Vec Psol, Vec sol, PS_SLL nBlockRows, PS_SLL iStart, PS_SLL nb);

// PETSc shell preconditioner
PetscErrorCode precondApplyMSP(PC pc, Vec xin, Vec xout);
PetscErrorCode precondApplyCPR(PC pc, Vec xin, Vec xout);
PS_SLL preSolver(Mat &A, Vec &b, Vec &x, bool zeroGuess, PS_SLL solverType);

// diagonal scaling (ABF)
void smat_inv_4x4(double *A);
void smat_identity_4x4(double *A);
void smat_mul_4x4(double *A, double *B, double *C);
void smat_vec_mul_4(double *A, double *b, double *c);
void decoup_abf_4x4(double *val, double *rhs, PS_SLL *rpt, PS_SLL *cpt, PS_SLL nb, PS_SLL nrow);

void smat_inv_2x2(double *A);
void smat_identity_2x2(double *A);
void smat_mul_2x2(double *A, double *B, double *C);
void smat_vec_mul_2(double *A, double *b, double *c);
void decoup_abf_2x2(double *val, double *rhs, PS_SLL *rpt, PS_SLL *cpt, PS_SLL nb, PS_SLL nrow);

void decouple_abf(double *val, double *rhs, PS_SLL *rpt, PS_SLL *cpt, PS_SLL nb, PS_SLL nrow);
void decoup_abf_nb(double *val, double *rhs, PS_SLL *rpt, PS_SLL *cpt, PS_SLL nb, PS_SLL nrow);

// Analytical decoupling method
void decouple_anl(double *val, double *rhs, PS_SLL *rpt, PS_SLL *cpt, PS_SLL nb, PS_SLL nrow, int is_thermal);
// Semi-analytical decoupling method
void decouple_sem(double *val, double *rhs, PS_SLL *rpt, PS_SLL *cpt, PS_SLL nb, PS_SLL nrow, int is_thermal);

// decoupling method
void decoup(double *val, double *rhs, PS_SLL *rpt, PS_SLL *cpt, PS_SLL nb, PS_SLL nrow, int decoup_type, int is_thermal);
#endif
