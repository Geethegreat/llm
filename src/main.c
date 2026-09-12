#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "matrix.h"
#include "activations.h"
#include "dataset.h"
#include "model.h"
#include "train.h"
#include "generate.h"

#define CONTEXT_SIZE 8
#define EMBEDDING_DIM 16
#define HIDDEN_DIM 128
#define TRAINING_STEPS 500000
#define DATA_PATH "../data/input.txt"
#define MODEL_PATH "../models/model.txt"

static void run_train(void){
    Dataset dataset = dataset_load(DATA_PATH);
    Model model = model_create(dataset.vocab_size, EMBEDDING_DIM, HIDDEN_DIM, CONTEXT_SIZE);

    train_model(&model, TRAINING_STEPS, &dataset);
    model_save(&model, MODEL_PATH, &dataset);

    printf("Training complete. Model saved to %s\n", MODEL_PATH);

    model_free(&model);
    dataset_free(&dataset);
}

static void run_generate(const char* prompt){
    Dataset dataset;
    Model model = model_load(MODEL_PATH, &dataset);

    generate_text(&model, &dataset, prompt, 100);

    model_free(&model);
    dataset_free(&dataset);
}

int main(int argc, char** argv){
    srand(time(NULL));

    if (argc < 2){
        fprintf(stderr, "Usage:\n  %s train\n  %s generate \"<prompt>\"\n", argv[0], argv[0]);
        return EXIT_FAILURE;
    }

    if (strcmp(argv[1], "train") == 0){
        run_train();
    } else if (strcmp(argv[1], "generate") == 0){
        if (argc < 3){
            fprintf(stderr, "generate requires a prompt, e.g. %s generate \"ROMEO\"\n", argv[0]);
            return EXIT_FAILURE;
        }
        run_generate(argv[2]);
    } else {
        fprintf(stderr, "Unknown command: %s\n", argv[1]);
        return EXIT_FAILURE;
    }

    return 0;
}