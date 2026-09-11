#ifndef TRAIN_H
#define TRAIN_H
#include "model.h"
#include "dataset.h"

void train_model(Model* model, int steps, Dataset* dataset);

#endif