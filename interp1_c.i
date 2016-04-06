 /* interp1_c.i */
 %module interp1_c
 %{
 /* Put header files here or function declarations like below */
extern float* interp1( float *yy, int yy_tam, float *x, int x_tam, float *y, int y_tam, float *xx, int xx_tam);
 %}
 
%{
#define SWIG_FILE_WITH_INIT
%}
%include "numpy.i"
%init %{
import_array();
%}

%apply (float* ARGOUT_ARRAY1, int DIM1) {(float *yy, int yy_tam)}
%apply (float* IN_ARRAY1, int DIM1) {(float *x, int x_tam)}
%apply (float* IN_ARRAY1, int DIM1) {(float *y, int y_tam)}
%apply (float* IN_ARRAY1, int DIM1) {(float *xx, int xx_tam)}


%include "interp1_c.h"
