// test_int.c

#define _GNU_SOURCE
#include <math.h>
#include <stdio.h>


void
test(double d, int expected) {
    double log = log10(fabs(d)) + 0.000022;
    double dexp = floor(log / 3.0);
    long exp = (long)dexp;
    double adj = exp10(dexp);
    printf("test(%f) log %f, dexp %f, exp %ld, got %ld, expected %d\n",
           d, log, dexp, exp, exp * 3, expected);
}


int
main() {
    printf("log10(9.9995) %lf\n", log10(9.9995));
    printf("log10(-9.9995) %lf\n", log10(-9.9995));
    printf("3.999 %d\n", (int)3.999);
    printf("-3.999 %d\n", (int)-3.999);
    printf("-3 %.0f\n", -3.0);
    test(0.00000099995, -6);
    test(0.0009999, -6);
    test(0.00099995, -3);
    test(0.9999, -3);
    test(0.0, 0);
    test(0.99995, 0);
    test(999.9, 0);
    test(999.95, 3);
    test(999900.0, 3);
    test(999950.0, 6);
    return 0;
}
