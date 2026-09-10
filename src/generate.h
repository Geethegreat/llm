#ifndef GENERATE_H
#define GENERATE_H

#include "model.h"
#include "dataset.h"

void generate_text( Model* model, Dataset* dataset, const char* prompt, int length);

#endif