https://huggingface.co/google-bert/bert-large-uncased

#### Pretraining

The model was trained on 4 cloud TPUs in Pod configuration (16 TPU chips total) for one million steps with a batch size of 256. The sequence length was limited to 128 tokens for 90% of the steps and 512 for the remaining 10%. The optimizer used is Adam with a learning rate of 1e-4, 𝛽1=0.9 and 𝛽2=0.999, a weight decay of 0.01, learning rate warmup for 10,000 steps and linear decay of the learning rate after.

#### Evaluation results
When fine-tuned on downstream tasks, this model achieves the following results:

Glue test results:

    SQUAD 1.1      Mulit NLI
    F1/EM          Accuracy
	91.0/84.3	   86.05