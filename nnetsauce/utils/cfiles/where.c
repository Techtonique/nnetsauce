//find indices of an element

// gcc -shared -Wl,-install_name,wherer.so -o wherer.so -fPIC where.c

#include <stdio.h>


/* function prototypes */
int where(double*, double, 
          long int, long int*);


/* functions' codes */
int where(double* x, double elt, 
          long int n, long int* res)
{
  for(unsigned long int i = 0; i < n; i++)
  {
    if (x[i] == elt)
    {
      res[i] = i; 
    } else {
      res[i] = -1;
    }
  }
  return(0);
}