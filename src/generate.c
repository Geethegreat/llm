#include <stdio.h>
#include <stdlib.h>
#include "model.h"
#include "dataset.h"
#include "string.h"
#define CONTEXT_SIZE 8
#define TEMPERATURE 1.0f


void generate_text(Model* model,Dataset* dataset,  const char* prompt, int length){
    int context_ids[CONTEXT_SIZE];
    int prompt_len = strlen(prompt);

    if(prompt_len < CONTEXT_SIZE){
        fprintf(stderr, " prompt must be atleast %d characters", CONTEXT_SIZE);
        return;
    }

    //encode the prompt
    for(int i =0;i < CONTEXT_SIZE;i++){
        context_ids[i]= dataset_encode(dataset, prompt[ prompt_len - CONTEXT_SIZE + i]);
    }

    printf("%s", prompt);

    for(int i =0 ;i<length;i++){
        int next_id = model_sample_temperature(model, context_ids, TEMPERATURE);
        char next_char = dataset_decode(dataset, next_id);

        printf("%c", next_char);

        //move context forward
        for(int j=0;j<CONTEXT_SIZE-1;j++){
            context_ids[j]=context_ids[j+1];
        }
        context_ids[CONTEXT_SIZE - 1]= next_id;
    }
printf("\n");
}