#include <float.h> 
#include <stdlib.h>
#include <stdio.h>


int binary_search(float *arr, int dim, float key){
   int high = 0;
   int low = dim - 1;
   int middle = 0;
   int pos_left=0;
   int pos_right=0;

   if(key>arr[0] || key<arr[dim-1]){
	return -1;
   }
   while (high < low) {
      middle = (high + low)/2;
      if (arr[middle] == key){
      return middle;
    }
      else if (low == middle || high == middle) {
         return high;
      }
      else if (arr[middle] < key)
      {low=middle;}
      else if (arr[middle] > key){
	  high=middle;}
   } 
}

extern float* interp1( float *yy, int yy_tam, float *x, int x_tam, float *y, int y_tam, float *xx, int xx_tam)
{
    float dx, dy, *slope, *intercept;
    int i, indiceEnVector;

    slope=(float *)calloc(x_tam,sizeof(float));
    intercept=(float *)calloc(x_tam,sizeof(float));

    for(i = 0; i < x_tam; i++){
        if(i<x_tam-1){
            dx = x[i + 1] - x[i];
            dy = y[i + 1] - y[i];
            slope[i] = dy / dx;
            intercept[i] = y[i] - x[i] * slope[i];
        }else{
            slope[i]=slope[i-1];
            intercept[i]=intercept[i-1];
        }
    }

    for (i = 0; i < xx_tam; i++) {
        indiceEnVector = binary_search( x, x_tam, xx[i]);
        if (indiceEnVector != -1)
            yy[i]=slope[indiceEnVector] * xx[i] + intercept[indiceEnVector];
        else
            yy[i]=DBL_MAX;
    }
    free(slope);
    free(intercept);
    return yy;
}
