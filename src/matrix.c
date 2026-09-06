#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"

Matrix matrix_create(int rows, int cols){
    Matrix m;

    m.rows = rows;
    m.cols= cols;

    m.data= malloc(rows * cols * sizeof(float));

    return m;
}

void matrix_free(Matrix* m){
    free(m->data);

    m->data=NULL;
    m->rows=0;
    m->cols=0;
}


void matrix_set(Matrix* m, int row, int col, int value){
    m->data[row * m->cols + col]=value;
}

float matrix_get(const Matrix* m, int row, int col){
    return m->data[row * m->cols +col];
}

void matrix_fill(Matrix* m, float value){
    for(int i=0;i<m->rows * m->cols;i++){
        m->data[i]=value;
    }
}

Matrix matrix_add(const Matrix* a ,const Matrix* b){
    Matrix res = matrix_create(a->rows,b->cols);

    for(int i=0;i<a->rows * a->cols;i++){
      res.data[i]=a->data[i]+b->data[i];
    }
    return res;
}

Matrix matrix_multiply(const Matrix* a ,const Matrix* b){
    Matrix res= matrix_create(a->rows,b->cols);

    matrix_fill(&res,0.0f);

    for(int i =0;i<a->rows;i++){
        for(int j=0;j<b->cols;j++){
            for(int k=0;k<a->cols;k++){
                res.data[i* res.cols + j] += a->data[i * a->cols + k] * b->data[k * b->cols + j]; 
            }
        }
    }
    return res;
}

Matrix matrix_transpose(const Matrix* m){
    Matrix res = matrix_create(m->cols, m->rows);

    for(int i=0;i<m->cols;i++){
        for(int j=0;j<m->rows;j++){
            matrix_set(&res, i, j, matrix_get(m, j,i));
        }
    }
    return res;
}


Matrix matrix_add_scalar(const Matrix* m, float value){
    Matrix res = matrix_create(m->rows, m->cols);

    for(int i =0;i<m->rows * m->cols;i++){
        res.data[i]=m->data[i]+value;
    }
    return res;
}

Matrix matrix_multiply_scalar(const Matrix* m, float value){
    Matrix res = matrix_create(m->rows, m->cols);

    for(int i =0;i<m->rows * m->cols;i++){
        res.data[i]=m->data[i] * value;
    }
    return res;
}


void matrix_print(const Matrix* m){
    for(int i =0;i<m->rows;i++){
        for(int j=0;j<m->cols;j++){
            printf("%f ", matrix_get(m, i, j));
        }
        printf("\n");
    }
}
