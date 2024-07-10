https://huggingface.co/FacebookAI/roberta-large

#### Pretraining

The model was trained on 1024 V100 GPUs for 500K steps with a batch size of 8K and a sequence length of 512. The optimizer used is Adam with a learning rate of 6e-4, 𝛽1=0.9, 𝛽2=0.98 𝜖=1𝑒−6, and a weight decay of 0.01, learning rate warmup for 30,000 steps and linear decay of the learning rate after.

#### Evaluation results
When fine-tuned on downstream tasks, this model achieves the following results:

Glue test results:

    mnli    qqp     qnli    sst2    cola    sstb    mrpc    rte 
	90.2	92.2	94.7	96.4	68.0	96.4	90.9	86.6