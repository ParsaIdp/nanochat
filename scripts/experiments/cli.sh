RUN_NAME=big-BPE TOKENIZER_NAME=tokenizer-big sbatch scripts/train_llm_tok.sh

RUN_NAME=normal-BPE TOKENIZER_NAME=tokenizer sbatch scripts/train_llm_tok.sh