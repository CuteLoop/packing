#include "export.h"
#include <stdio.h>
#include <math.h>

void export_svg(const char *filename,
                const ConvexDecomp *D,
                const Pose *poses, int nInstances,
                Container container,
                double scale)
{
    FILE *f = fopen(filename, "w");
    if (!f) {
        printf("Error: Could not write to %s\n", filename);
        return;
    }

    double width_px = container.width * scale;
    double height_px = container.height * scale;
    double margin = 50.0;

    fprintf(f, "<svg xmlns='http://www.w3.org/2000/svg' "
               "width='%.0f' height='%.0f' "
               "viewBox='%.0f %.0f %.0f %.0f'>\n",
            width_px + margin*2, height_px + margin*2,
            -width_px/2 - margin, -height_px/2 - margin,
             width_px + margin*2,  height_px + margin*2);

    fprintf(f, "<rect x='%.2f' y='%.2f' width='%.2f' height='%.2f' "
               "fill='none' stroke='black' stroke-width='2'/>\n",
            -container.width*0.5*scale, -container.height*0.5*scale,
             container.width*scale,      container.height*scale);

    const char *colors[] = {"#e74c3c", "#3498db", "#2ecc71", "#f1c40f", "#9b59b6"};

    for (int i = 0; i < nInstances; i++) {
        Pose p = poses[i];
        double c = cos(p.ang);
        double s = sin(p.ang);
        const char *color = colors[i % 5];

        for (int k = 0; k < D->nParts; k++) {
            ConvexPart *part = &D->parts[k];
            fprintf(f, "<polygon points='");
            for (int v = 0; v < part->n; v++) {
                double lx = part->v[v].x;
                double ly = part->v[v].y;
                double wx = (c * lx - s * ly) + p.t.x;
                double wy = (s * lx + c * ly) + p.t.y;
                fprintf(f, "%.2f,%.2f ", wx * scale, wy * scale);
            }
            fprintf(f, "' fill='%s' stroke='black' stroke-width='1' opacity='0.8' />\n", color);
        }
    }

    fprintf(f, "</svg>\n");
    fclose(f);
    printf("Exported visualization to %s\n", filename);
}
