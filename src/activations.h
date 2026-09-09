#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H
#include "matrix.h"

Matrix relu(const Matrix* m);

Matrix softmax(const Matrix* m);


Matrix softmax_cross_entropy_backward(const Matrix* pred, Matrix* target);

float cross_entropy_loss(const Matrix* pred, const Matrix* target);


#endif