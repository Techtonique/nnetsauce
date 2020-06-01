/* Authors: T. Moudiki <thierry.moudiki@gmail.com>

 License: BSD 3 Clause Clear */

//check if vector is a factor

// gcc -shared -Wl,-install_name,check_factorer.so -o check_factorer.so -fPIC check_factor.c

#include <stdio.h>
#include <math.h>


/* function prototypes */
int check_factor(double*, long int);


/* functions' codes */
int check_factor(double* x, long int n)
{
  int res = 1;
  double diff = 0;
  for(long int i = 0; i < n; i++)
  {
    diff = x[i] - round(x[i]);
    if(diff != 0.0)
    {
      return(0);
    }
  }
  return(res);
}