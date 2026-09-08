#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include "dataset.h"
#include <time.h>
#include <string.h>
#include <stdbool.h>


Dataset dataset_create(const char* text, const char* vocab){
    Dataset dataset;

    dataset.text_length = strlen(text);
    dataset.text = malloc(dataset.text_length * sizeof(char));
    memcpy(dataset.text, text, dataset.text_length);

    dataset.vocab_size = strlen(vocab);
    dataset.vocab = malloc(dataset.vocab_size * sizeof(char));
    memcpy(dataset.vocab, vocab, dataset.vocab_size);

    return dataset;
}

Dataset dataset_load(const char* filename){
    Dataset dataset;
    FILE* file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    fseek(file, 0, SEEK_END);
    dataset.text_length = ftell(file);
    rewind(file);

    dataset.text = malloc(dataset.text_length * sizeof(char));
    fread(dataset.text, sizeof(char), dataset.text_length, file);
    fclose(file);

    bool seen[256] = {false};
    dataset.vocab = malloc(256 * sizeof(char));
    int vocab_index = 0;
    for (int i = 0; i < dataset.text_length; i++) {
        unsigned char c = dataset.text[i];
        if (!seen[c]) { seen[c] = true; dataset.vocab[vocab_index++] = c; }
    }
    dataset.vocab_size = vocab_index;

    return dataset;
}

int dataset_encode(const Dataset* dataset, char c){
    for(int i=0;i<dataset->vocab_size;i++){
        if(dataset->vocab[i]==c){
            return i;
        }
    }
    return -1;
}

char dataset_decode(const Dataset* dataset, int index){
    return dataset->vocab[index];   
}

void dataset_sample(const Dataset* dataset, int context_length, int* context_ids_out, int* target_id_out){
    int start = rand()  % (dataset->text_length - context_length -1);

    for(int i=0;i<context_length;i++){
        context_ids_out[i]= dataset_encode(dataset, dataset->text[start + i]);
    }
    *target_id_out = dataset_encode(dataset, dataset->text[start + context_length]);
}

void dataset_free(Dataset* dataset){
    free(dataset->text);
    free(dataset->vocab);
    dataset->text_length=0;
    dataset->vocab_size=0;
    free(dataset);
}
