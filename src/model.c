#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

void model_forward(const Model* model, int* context_ids, ForwardCache* cache){
        Matrix context_embeddings = matrix_create(model->context_size, model->embedding_dim);

        for(int i=0;i<model->context_size;i++){
            for(int j=0;j<model->embedding_dim;j++){
                context_embeddings.data[i * model->embedding_dim + j] = model->C.data[context_ids[i] * model->embedding_dim + j];
            }
        }

        Matrix flattened_context_embeddings = matrix_flatten(&context_embeddings);
        cache->X = flattened_context_embeddings;
        
        matrix_free(&context_embeddings);
        
        Matrix hidden_layer = matrix_multiply(&flattened_context_embeddings, &model->W1);

        Matrix hidden_layer_with_bias = matrix_add(&hidden_layer, &model->B1);
        cache->Z1=hidden_layer_with_bias;

        Matrix activated_hidden_layer = relu(&hidden_layer_with_bias);
        cache->H = activated_hidden_layer;

        Matrix output_layer = matrix_multiply(&activated_hidden_layer, &model->W2);

        Matrix output_layer_with_bias = matrix_add(&output_layer, &model->B2);
        matrix_free(&hidden_layer);
        
        Matrix softmaxed_output = softmax(&output_layer_with_bias);
        cache->probs = softmaxed_output;
}

Gradients model_backward(Model* model, ForwardCache* cache,  int* context_ids, int target_id){
        Gradients grad;

        Matrix y = matrix_create(1, model->vocab_size);
        matrix_fill(&y, 0.0f);
        y.data[target_id] = 1.0f;

        Matrix delta_out = matrix_substract(&cache->probs, &y);

        Matrix H_T = matrix_transpose(&cache->H);

        grad.dW2 = matrix_multiply(&H_T, &delta_out);
        matrix_free(&H_T);

        grad.dB2 = matrix_create(1, model->vocab_size);

        for(int i=0;i<model->vocab_size;i++){
            grad.dB2.data[i] = delta_out.data[i];
        
        }

        Matrix W2_T = matrix_transpose(&model->W2);
        Matrix dH = matrix_multiply(&delta_out,&W2_T);

        Matrix delta_hidden = matrix_create(1, model->hidden_dim);

        for(int i =0;i<model->hidden_dim;i++){
            delta_hidden.data[i]=(cache->H.data[i]>0) ? dH.data[i] : 0.0f;
        }
        
        Matrix X_T = matrix_transpose(&cache->X);

        grad.dW1 = matrix_multiply(&X_T,&delta_hidden);
        
        grad.dB1 = matrix_create(1, model->hidden_dim);

        for(int i=0;i<model->hidden_dim;i++){
            grad.dB1.data[i] = delta_hidden.data[i];
        
        }

        Matrix W1_T = matrix_transpose(&model->W1);
        Matrix dX = matrix_multiply(&delta_hidden, &W1_T);

        grad.dC = matrix_create(model->vocab_size, model->embedding_dim);   
        matrix_fill(&grad.dC, 0.0f);

        for(int i =0;i<model->context_size;i++){
            int char_id = context_ids[i];
            for(int j=0;j<model->embedding_dim;j++){
                grad.dC.data[char_id * model->embedding_dim + j] += dX.data[i*model->embedding_dim+j];
            }
        }
        matrix_free(&y);
        matrix_free(&delta_out);
        matrix_free(&W2_T);
        matrix_free(&dH);
        matrix_free(&delta_hidden);
        matrix_free(&X_T);
        matrix_free(&W1_T);
        matrix_free(&dX);
        return grad;
}

float numerical_gradient_check(Model* model, float* param, int* context_ids, int target_id) {
    float epsilon = 1e-4f;
    float original = *param;

    *param = original + epsilon;
    ForwardCache cache_plus;
    model_forward(model, context_ids, &cache_plus);
    float loss_plus = model_loss(&cache_plus.probs, target_id);

    *param = original - epsilon;
    ForwardCache cache_minus;
    model_forward(model, context_ids, &cache_minus);
    float loss_minus = model_loss(&cache_minus.probs, target_id);

    *param = original;  
    return (loss_plus - loss_minus) / (2.0f * epsilon);
}

void model_update(Model* model, Gradients* grad, float lr){
    for(int i=0; i<model->C.rows*model->C.cols; i++)  model->C.data[i]  -= lr * grad->dC.data[i];
    for(int i=0; i<model->W1.rows*model->W1.cols; i++) model->W1.data[i] -= lr * grad->dW1.data[i];
    for(int i=0; i<model->B1.rows*model->B1.cols; i++) model->B1.data[i] -= lr * grad->dB1.data[i];
    for(int i=0; i<model->W2.rows*model->W2.cols; i++) model->W2.data[i] -= lr * grad->dW2.data[i];
    for(int i=0; i<model->B2.rows*model->B2.cols; i++) model->B2.data[i] -= lr * grad->dB2.data[i];
}

// greedy decoding
int model_predict(const Model* model, int* context_ids){
    ForwardCache cache;

    model_forward(model, context_ids, &cache);

    //finding the highest probability of all probs
    int best_id = 0;
    float highest_probs = cache.probs.data[0];

    for(int i =1;i<model->vocab_size;i++){
        if(cache.probs.data[i]>highest_probs){
            highest_probs = cache.probs.data[i];
            best_id = i;
        }
    }
    return best_id;
}

// as greedy decoding results in repetition use random sampling
int model_sample(const Model* model, int* context_ids){
    ForwardCache cache;
    model_forward(model, context_ids, &cache);

    float r = (float)rand() / (float)RAND_MAX;
    float cumulative = 0.0f;
    int chosen =model->vocab_size-1;

    for(int i =0;i<model->vocab_size;i++){
        cumulative += cache.probs.data[i];
        if(r < cumulative){
            chosen = i;
            break;
        }
    }
    
}

void model_save(const Model* model, const char* filename, Dataset* dataset){
    FILE* f = fopen(filename, "wb");
    if(!f){fprintf(stderr, "could not open %s for writing\n", filename); exit(EXIT_FAILURE);}

    fwrite(&model->vocab_size, sizeof(int), 1, f);
    fwrite(&model->embedding_dim, sizeof(int), 1, f);
    fwrite(&model->hidden_dim, sizeof(int), 1, f);
    fwrite(&model->context_size, sizeof(int), 1, f);

    fwrite(model->C.data,  sizeof(float), model->C.rows  * model->C.cols,  f);
    fwrite(model->W1.data, sizeof(float), model->W1.rows * model->W1.cols, f);
    fwrite(model->B1.data, sizeof(float), model->B1.rows * model->B1.cols, f);
    fwrite(model->W2.data, sizeof(float), model->W2.rows * model->W2.cols, f);
    fwrite(model->B2.data, sizeof(float), model->B2.rows * model->B2.cols, f);

    fwrite(dataset->vocab, sizeof(char), dataset->vocab_size, f);
    fclose(f);
}

Model model_load(const char* filename, Dataset* dataset_out){
    FILE* f = fopen(filename, "rb");
    if(!f){fprintf(stderr, "could not open %s for reading\n", filename); exit(EXIT_FAILURE);}

    int vocab_size, embedding_dim, hidden_dim, context_size;
    fread(&vocab_size, sizeof(int), 1, f);
    fread(&embedding_dim, sizeof(int), 1, f);
    fread(&hidden_dim, sizeof(int), 1, f);
    fread(&context_size, sizeof(int), 1, f);

    Model model = model_create(vocab_size, embedding_dim, hidden_dim, context_size);

    fread(model.C.data,  sizeof(float), model.C.rows  * model.C.cols,  f);
    fread(model.W1.data, sizeof(float), model.W1.rows * model.W1.cols, f);
    fread(model.B1.data, sizeof(float), model.B1.rows * model.B1.cols, f);
    fread(model.W2.data, sizeof(float), model.W2.rows * model.W2.cols, f);
    fread(model.B2.data, sizeof(float), model.B2.rows * model.B2.cols, f);

    dataset_out->vocab_size = vocab_size;
    dataset_out->vocab = malloc(vocab_size * sizeof(char));
    fread(dataset_out->vocab, sizeof(char), vocab_size, f);
    dataset_out->text = NULL;      // not needed for generation, only training
    dataset_out->text_length = 0;

    fclose(f);
    return model;
}

float model_loss(const Matrix* pred, int target){
    float p = pred->data[target];
    return -logf(p + 1e-9f);
}

