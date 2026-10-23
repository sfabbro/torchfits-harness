/*
 * dump_cfitsio_values -- independent value oracle for the harness.
 *
 * Reads values with CFITSIO only (no astropy, no torchfits) and prints them as
 * text so the Python test can diff them against torchfits:
 *
 *   dump_cfitsio_values image <file> <hdu> <x0> <y0> <nx> <ny>
 *   dump_cfitsio_values table <file> <ext> <column> <firstrow> <nrows>
 *   dump_cfitsio_values info  <file> <hdu>
 *
 * Coordinates are zero-based, matching fits_read_subset. Rows are zero-based,
 * matching the Python test's row windows.
 *
 * Output: a header line beginning with '#' and then one value per line,
 * 'nan' for nulls, %.17g otherwise.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fitsio.h"

static int report(int status, const char *what) {
    if (status == 0) {
        return 0;
    }
    char message[FLEN_ERRMSG];
    fits_get_errstatus(status, message);
    fprintf(stderr, "%s: %s (%d)\n", what, message, status);
    return 1;
}

static void print_value(double value, int null) {
    if (null || isnan(value)) {
        printf("nan\n");
    } else {
        printf("%.17g\n", value);
    }
}

static int cmd_info(const char *path, int hdu) {
    fitsfile *fptr = NULL;
    int status = 0;
    if (fits_open_file(&fptr, path, READONLY, &status)) {
        return report(status, "open");
    }
    int bitpix = 0, naxis = 0;
    long naxes[9] = {0};
    if (fits_get_img_param(fptr, 9, &bitpix, &naxis, naxes, &status)) {
        return report(status, "img_param");
    }
    printf("# bitpix %d naxis %d", bitpix, naxis);
    for (int i = 0; i < naxis; i++) {
        printf(" %ld", naxes[i]);
    }
    printf("\n");
    fits_close_file(fptr, &status);
    return report(status, "close");
}

static int cmd_image(const char *path, int hdu, long x0, long y0, long nx, long ny) {
    fitsfile *fptr = NULL;
    int status = 0;
    if (fits_open_file(&fptr, path, READONLY, &status)) {
        return report(status, "open");
    }
    if (fits_movabs_hdu(fptr, hdu + 1, NULL, &status)) {
        return report(status, "movabs_hdu");
    }
    long naxes[9] = {0};
    int naxis = 0;
    if (fits_get_img_param(fptr, 9, NULL, &naxis, naxes, &status)) {
        return report(status, "img_param");
    }
    if (naxis < 2) {
        fprintf(stderr, "image: expected at least 2 axes, got %d\n", naxis);
        return 2;
    }
    if (x0 < 0 || y0 < 0 || x0 + nx > naxes[0] || y0 + ny > naxes[1]) {
        fprintf(stderr, "image: window outside %ldx%ld\n", naxes[0], naxes[1]);
        return 2;
    }

    long fpixel[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    long lpixel[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    long inc[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    fpixel[0] = x0 + 1;
    fpixel[1] = y0 + 1;
    lpixel[0] = x0 + nx;
    lpixel[1] = y0 + ny;
    /* Full range on any further axes. */
    for (int i = 2; i < naxis; i++) {
        fpixel[i] = 1;
        lpixel[i] = naxes[i];
    }

    size_t count = (size_t)nx * (size_t)ny;
    for (int i = 2; i < naxis; i++) {
        count *= (size_t)naxes[i];
    }
    double *buffer = malloc(count * sizeof(double));
    if (buffer == NULL) {
        fprintf(stderr, "image: out of memory\n");
        return 2;
    }
    int anynull = 0;
    double nulval = NAN;
    if (fits_read_subset(fptr, TDOUBLE, fpixel, lpixel, inc, &nulval, buffer,
                         &anynull, &status)) {
        free(buffer);
        return report(status, "read_subset");
    }

    printf("# shape %ld %ld", ny, nx);
    for (int i = 2; i < naxis; i++) {
        printf(" %ld", naxes[i]);
    }
    printf(" anynull %d\n", anynull);
    for (size_t i = 0; i < count; i++) {
        print_value(buffer[i], 0);
    }
    free(buffer);
    fits_close_file(fptr, &status);
    return report(status, "close");
}

static int cmd_table(const char *path, int ext, const char *column, long firstrow,
                     long nrows) {
    fitsfile *fptr = NULL;
    int status = 0;
    if (fits_open_file(&fptr, path, READONLY, &status)) {
        return report(status, "open");
    }
    if (fits_movabs_hdu(fptr, ext + 1, NULL, &status)) {
        return report(status, "movabs_hdu");
    }
    int colnum = 0;
    if (fits_get_colnum(fptr, CASESEN, (char *)column, &colnum, &status)) {
        return report(status, "get_colnum");
    }
    long rowlen = 0;
    int typecode = 0;
    long repeat = 0;
    long width = 0;
    if (fits_get_coltype(fptr, colnum, &typecode, &repeat, &width, &status)) {
        return report(status, "get_coltype");
    }
    long total = repeat * nrows;
    double *buffer = malloc((size_t)total * sizeof(double));
    if (buffer == NULL) {
        fprintf(stderr, "table: out of memory\n");
        return 2;
    }
    int anynull = 0;
    if (fits_read_col(fptr, TDOUBLE, colnum, firstrow + 1, 1, total, NULL, buffer,
                      &anynull, &status)) {
        free(buffer);
        return report(status, "read_col");
    }
    printf("# column %s typecode %d repeat %ld rows %ld anynull %d\n", column,
           typecode, repeat, nrows, anynull);
    for (long i = 0; i < total; i++) {
        print_value(buffer[i], 0);
    }
    free(buffer);
    fits_close_file(fptr, &status);
    return report(status, "close");
}

static void usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s image <file> <hdu> <x0> <y0> <nx> <ny>\n"
            "       %s table <file> <ext> <column> <firstrow> <nrows>\n"
            "       %s info  <file> <hdu>\n",
            argv0, argv0, argv0);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        usage(argv[0]);
        return 2;
    }
    if (strcmp(argv[1], "info") == 0 && argc == 4) {
        return cmd_info(argv[2], atoi(argv[3]));
    }
    if (strcmp(argv[1], "image") == 0 && argc == 8) {
        return cmd_image(argv[2], atoi(argv[3]), atol(argv[4]), atol(argv[5]),
                         atol(argv[6]), atol(argv[7]));
    }
    if (strcmp(argv[1], "table") == 0 && argc == 7) {
        return cmd_table(argv[2], atoi(argv[3]), argv[4], atol(argv[5]),
                         atol(argv[6]));
    }
    usage(argv[0]);
    return 2;
}
