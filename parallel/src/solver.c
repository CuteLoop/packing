// Backwards-compatibility wrapper. Calls the serial solver implementation.
#include "solver.h"

// If other code still calls run_solver, forward to the serial implementation.
void run_solver(const ConvexDecomp *D, Pose *poses, int nInstances,
                Container *container, SolverParams params)
{
    run_solver_serial(D, poses, nInstances, container, params);
}
