#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix.h"

Matrix relu(const Matrix* m){
    Matrix res = matrix_create(m->rows, m->cols);

    for(int i=0;i<m->rows * m-> cols;i++){
        if(m->data[i]>=0.0f){
            res.data[i]=m->data[i];
        }
        else{
            res.data[i]=0.0f;
        }
    }
    return res;
}

Matrix softmax(const Matrix* m){
        Matrix res = matrix_create(m->rows, m->cols);
        float sum =0.0f;

        for(int i=0;i<m->rows * m-> cols;i++){
            res.data[i]= expf(m->data[i]);
            sum+=res.data[i];            
        }

      for(int i=0;i<m->rows * m-> cols;i++){
        res.data[i]/=sum;
      }
    return res;
}

Matrix softmax_cross_entropy_backward(const Matrix* pred, Matrix* target){
    Matrix res = matrix_create(pred->rows, pred->cols);

    for(int i=0;i<pred->rows * pred-> cols;i++){
        res.data[i]=pred->data[i]-target->data[i];
    }
    return res;
}

