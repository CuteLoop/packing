#ifndef EXPORT_H
#define EXPORT_H

#include "convex_decomp.h"
#include "energy.h" // For Container struct

// Export the final state to an SVG file.
// filename: "output.svg"
// scale: Pixels per unit (e.g., 10.0 or 20.0)
void export_svg(const char *filename,
                const ConvexDecomp *D,
                const Pose *poses, int nInstances,
                Container container,
                double scale);

#endif
