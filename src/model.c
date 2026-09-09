#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include "activations.h"
#include "dataset.h"
#include "model.h"
#include <time.h>
#include <math.h>


Model model_create(int vocab_size, int embedding_dim, int hidden_dim, int context_size){
    Model model;
    model.vocab_size = vocab_size;
    model.embedding_dim = embedding_dim;   
    model.hidden_dim = hidden_dim;
    model.context_size = context_size;

    model.C = matrix_create(vocab_size, embedding_dim);
    matrix_fill_random(&model.C, -0.1f, 0.1f);

    model.W1 = matrix_create(context_size * embedding_dim, hidden_dim);
    matrix_fill_random(&model.W1, -0.1f, 0.1f);

    model.B1 = matrix_create(1, hidden_dim);
    matrix_fill(&model.B1, 0.0f);

    model.W2= matrix_create(hidden_dim, vocab_size);
    matrix_fill_random(&model.W2, -0.1f, 0.1f);

    model.B2 = matrix_create(1, vocab_size);
    matrix_fill_random(&model.B2, -0.1f, 0.1f);

    return model;
}

Matrix model_forward(const Model* model, int* context_ids){
        Matrix context_embeddings = matrix_create(model->context_size, model->embedding_dim);

        for(int i=0;i<model->context_size;i++){
            for(int j=0;j<model->embedding_dim;j++){
                context_embeddings.data[i * model->embedding_dim + j] = model->C.data[context_ids[i] * model->embedding_dim + j];
            }
        }

        Matrix flattened_context_embeddings = matrix_flatten(&context_embeddings);

        Matrix hidden_layer = matrix_multiply(&flattened_context_embeddings, &model->W1);

        Matrix hidden_layer_with_bias = matrix_add(&hidden_layer, &model->B1);

        Matrix activated_hidden_layer = relu(&hidden_layer_with_bias);

        Matrix output_layer = matrix_multiply(&activated_hidden_layer, &model->W2);

        Matrix output_layer_with_bias = matrix_add(&output_layer, &model->B2);

        Matrix softmaxed_output = softmax(&output_layer_with_bias);

        return softmaxed_output;
}


float model_loss(const Matrix* pred, int target){
    float p = pred->data[target];
    return -logf(p + 1e-9f);
}

