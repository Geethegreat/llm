#ifndef MATRIX_H
#define MATRIX_H

typedef struct{
    int rows;
    int cols;
    float* data;
} Matrix;


Matrix matrix_create(int rows, int cols);
void matrix_free(Matrix* m);

void matrix_set(Matrix* m, int row, int col, float value);
float matrix_get(const Matrix* m, int row, int col);
void matrix_fill_random(Matrix* m, float min, float max);
void matrix_fill(Matrix* m, float value);



Matrix matrix_add(const Matrix* a ,const Matrix* b);
Matrix matrix_multiply(const Matrix* a ,const Matrix* b);
Matrix matrix_transpose(const Matrix* m);
Matrix matrix_flatten(Matrix* m);

Matrix matrix_add_scalar(const Matrix* m, float value);
Matrix matrix_multiply_scalar(const Matrix* m, float value);

void matrix_print(const Matrix* m);

#endif