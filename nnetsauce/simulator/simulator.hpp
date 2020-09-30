# include <string>
# include <cstring>

// sobol
int i4_bit_hi1 (int n);
int i4_bit_lo0 (int n);
int i4_max (int i1, int i2);
int i4_min (int i1, int i2);
void i4_sobol (int dim_num, int *seed, float quasi[ ]);
float *i4_sobol_generate (int m, int n, int skip);

void i4_sobol_generate_in_C(float r[], int m, int n, int skip);

int i4_uniform (int b, int c, int *seed);

int i8_bit_hi1 (long long int n);
int i8_bit_lo0 (long long int n);
long long int i8_max (long long int i1, long long int i2);
long long int i8_min (long long int i1, long long int i2);
void i8_sobol (int dim_num, long long int *seed, double quasi[ ]);
double *i8_sobol_generate (int m, int n, int skip);
long long int i8_uniform (long long int b, long long int c, int *seed);

float r4_abs (float x);
int r4_nint (float x);
float r4_uniform_01 (int *seed);

double r8_abs (double x);
int r8_nint (double x);
double r8_uniform_01 (int *seed);

void r8mat_write (std::string output_filename, int m, int n, double table[]);

int tau_sobol (int dim_num);
void timestamp ();


// halton
double *halton (int i, int m);
double *halton_base (int i, int m, int b[]);
int halton_inverse (double r[], int m);
double *halton_sequence (int i1, int i2, int m);
int i4vec_sum (int n, int a[]);
int prime (int n);
double r8_mod (double x, double y);
void r8mat_print (int m, int n, double a[], std::string title);
void r8mat_print_some (int m, int n, double a[], int ilo, int jlo, int ihi,
  int jhi, std::string title);


// hammersley
double *hammersley (int i, int m, int n);
int hammersley_inverse (double r[], int m, int n);
double *hammersley_sequence (int i1, int i2, int m, int n);

