#ifndef MODEL_H
#define MODEL_H
#include "matrix.h"
#include "dataset.h"

typedef struct{
    Matrix C; // Embedding matrix/ vocabulary matrix
    Matrix W1;
    Matrix B1;
    Matrix W2;
    Matrix B2;

    int vocab_size, embedding_dim, hidden_dim, context_size;
} Model;

typedef struct{
    Matrix X;  //flattened context embeddings
    Matrix Z1;
    Matrix H; //hidden layer after linear transformation
    Matrix probs; //output probabilities after softmax
} ForwardCache;

typedef struct {
    Matrix dC, dW1, dB1, dW2, dB2;
} Gradients;


Model model_create(int vocab_size, int embedding_dim, int hidden_dim, int context_size);
 
void model_forward(const Model* model, int* context_ids, ForwardCache* cache);

Gradients model_backward(Model* model, ForwardCache* cache,  int* context_ids, int target_id);

void model_update(Model* model, Gradients* grad, float lr);

Model model_load(const char* filename, Dataset* dataset_out);

void model_save(const Model* model, const char* filename, Dataset* dataset);

float model_loss(const Matrix* pred, int target);

float numerical_gradient_check(Model* model, float* param, int* context_ids, int target_id);

int model_predict(const Model* model, int* context_ids);

void model_free(Model* model);


#endif


