#ifndef MODEL_H
#define MODEL_H
#include "matrix.h"

typedef struct{
    Matrix C; // Embedding matrix/ vocabulary matrix
    Matrix W1;
    Matrix B1;
    Matrix W2;
    Matrix B2;

    int vocab_size, embedding_dim, hidden_dim, context_size;
} Model;


Model model_create(int vocab_size, int embedding_dim, int hidden_dim, int context_size);
 
Matrix model_forward(const Model* model, int* context_ids);

Matrix model_backward(Model* model, int* context_ids, int target_id, float learning_rate);

void model_update(Model* model, Matrix gradiens, float learning_rate);

void model_save(const Model* model, const char* filename);

float model_loss(const Matrix* pred, int target);

void model_free(Model* model);


#endif


