markdown from 

## A. Dataset and Task

### Sentiment Analysis (SA)
- **Description**: Classifying sentiment polarities for a given sentence.
- **Datasets**: IMDB, SST-2, laptop, restaurant, Movie, Tweet.

### Natural Language Inference (NLI)
- **Description**: Determining whether a hypothesis is entailed or contradicted by a given sentence.
- **Datasets**: MNLI, QNLI, SNLI.

### Ethical Judgment (EJ)
- **Description**: Deciding whether a sentence is ethically acceptable.
- **Datasets**: Ethics/deontology, Ethics/justice.

### Paraphrase Identification (PI)
- **Description**: Classifying whether a pair of sentences are semantically identical.
- **Datasets**: QQP, MRPC.

### Question Answering (QA)
- **Description**: Answering a question based on given content.
- **Datasets**: SQuAD, NQ-Open.

### Summarization (SUM)
- **Description**: Summarizing a given article and generating the abstract.
- **Datasets**: Multi-News, SAMSum.

## A.2 Evaluation Metrics
- **Classification Tasks (SA, NLI, EJ, PI)**: Accuracy (Acc.)
- **Generation Tasks (QA, SUM)**: F1 and ROUGE-L.

## A.3 Prompt Tuning Setting
- **Optimizer**: AdamW
- **Learning Rate**: 0.001
- **Soft Prompts Length**: 100
- **Soft Prompts Initialization and Optimization**: Random initialization and predicts the label tokens at the [MASK] position.

## A.4 Label Tokens
- **SA (Sentiment Analysis)**: 
  - IMDB: positive, negative
  - SST-2: positive, negative
  - laptop: positive, moderate, negative
  - restaurant: positive, moderate, negative
  - Movie: positive, negative
  - Tweet: positive, moderate, negative

- **NLI (Natural Language Inference)**: 
  - MNLI: yes, neutral, no
  - QNLI: yes, no
  - SNLI: yes, neutral, no

- **EJ (Ethical Judgment)**:
  - deontology: acceptable, unacceptable
  - justice: acceptable, unacceptable

- **PI (Paraphrase Identification)**:
  - QQP: true, false
  - MRPC: true, false

## B. Cross-Task Transfer

### B.1 More Zero-shot Transfer Performance
- **Zero-shot Transfer Performance Investigation**: Analyzing performance on different sizes of RoBERTa and T5.


