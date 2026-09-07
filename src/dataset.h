#ifndef DATASET_H
#define DATASET_H

typedef struct {
    char* text;
    int text_length;

    char* vocab;
    int vocab_size;
}Dataset;


Dataset dataset_create(const char*text, const char* vocab);
Dataset dataset_load(const char* filename);

int dataset_encode(const Dataset* dataset, char c);
char dataset_decode(const Dataset* dataset, int index);

void dataset_sample(const Dataset* dataset, int context_length, int* context_ids_out, int* target_id_out);

void dataset_free(Dataset* dataset);

#endif
