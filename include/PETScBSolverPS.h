#ifndef PETScBSolverPS_h
#define PETScBSolverPS_h

// #ifdef __cplusplus
// extern "C" {
// #endif

#include "PETScSolver.h"

// #ifdef __cplusplus
// }
// #endif

// #ifdef __cplusplus
// extern "C" {
// #endif

class PETScBSolverPS
{
public:
    dBSRmat_ Absr;
    dvector_ b;
    dvector_ x;

    void allocate(PS_SLL row, PS_SLL col, PS_SLL nb, PS_SLL nnz, PS_SLL storage_manner);
    void destroy();
    PS_SLL petscsolve();
};

#ifdef __cplusplus
extern "C"
{
#endif

    PS_SLL FIM_solver(const PS_SLL commRoot, PS_SLL myid, PS_SLL num_procs, PS_SLL nrow, PS_SLL nb, PS_SLL *rpt, PS_SLL *cpt, double *val, double *rhs, double *sol);

    PS_SLL FIM_solver_p(int precondID, int is_thermal, int myid, int num_procs, int nb, PS_SLL *allLower, PS_SLL *allUpper, PS_SLL *rpt, PS_SLL *cpt, double *val, double *rhs, double *sol);

    PS_SLL FIM_solver_p_cpr(int is_thermal, int myid, int num_procs, int nb, PS_SLL *allLower, PS_SLL *allUpper, PS_SLL *rpt, PS_SLL *cpt, double *val, double *rhs, double *sol);

    PS_SLL FIM_solver_p_msp(int is_thermal, int myid, int num_procs, int nb, PS_SLL *allLower, PS_SLL *allUpper, PS_SLL *rpt, PS_SLL *cpt, double *val, double *rhs, double *sol);

    PS_SLL FIM_solver_p_bamg(int is_thermal, int myid, int num_procs, int nb, PS_SLL *allLower, PS_SLL *allUpper, PS_SLL *rpt, PS_SLL *cpt, double *val, double *rhs, double *sol);

    PS_SLL fim_solver_(const PS_SLL *commRoot, PS_SLL *myid, PS_SLL *num_procs, PS_SLL *nrow, PS_SLL *nb, PS_SLL *rpt, PS_SLL *cpt, double *val, double *rhs, double *sol);

#ifdef __cplusplus
}
#endif

#endif
