#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include "activations.h"
#include "dataset.h"
#include "model.h"
#define LEARNING_RATE 0.01f
#define CONTEXT_SIZE 8


void train_model(Model* model, int steps, Dataset* dataset){
    int context_ids[CONTEXT_SIZE];
    int target_id;
    float running_loss = 0.0f;


    for(int i=0;i<steps;i++){
        dataset_sample(dataset, CONTEXT_SIZE, context_ids, &target_id);
        ForwardCache cache;
        model_forward(model, context_ids, &cache);
        float loss = model_loss(&cache.probs, target_id);
        running_loss += loss;
        Gradients grad = model_backward(model, &cache, context_ids, target_id);
        model_update(model, &grad, LEARNING_RATE);

        matrix_free(&cache.X); matrix_free(&cache.H); matrix_free(&cache.probs);
        matrix_free(&grad.dC); matrix_free(&grad.dW1); matrix_free(&grad.dB1); matrix_free(&grad.dW2); matrix_free(&grad.dB2);
        
        if(i % 2000 == 0 && i > 0){
              printf("step %d, avg loss: %f\n", i, running_loss / 2000.0f);
            running_loss = 0.0f;
        }
    }

}