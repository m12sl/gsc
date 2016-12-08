# Классификация Геномных транскрипций

Пытаемся определить, как можно различить кодирующие последовательности от не кодирующих.



## Random Short Slice CNN

Режем последовательности на случайные фрагменты небольшой длины и тренируем сверточную сеть.

В папке `./data` должно лежать два файла:
`gencode.v19.pc_transcripts.fa  lncipedia_3_1.fasta`


Для тренировки запускаем скрипт `train.py` с параметрами:
```
usage: train.py [-h] [--data DATA] [--save_dir SAVE_DIR]
                [--vocab_size VOCAB_SIZE] [--dim DIM]
                [--batch_size BATCH_SIZE] [--seq_length SEQ_LENGTH]
                [--steps STEPS] [--epochs EPOCHS] [--keep_prob KEEP_PROB]
                [--learning_rate LEARNING_RATE] [--seed SEED]
                [--outer_split OUTER_SPLIT] [--inner_split INNER_SPLIT]
```

Например
```
CUDA_VISIBLE_DEVICES=0 python train.py --keep_prob 0.5 \
                    --batch_size 512 --seq_length=1024 \
                    --epochs 100 --save_dir tmp/cnn-1024-kp0.5
```


После работы в папке `save_dir` будут файлы модельки и история обучения `history.log`.

### TODO:

- Написать инференс, надо визуализировать на что она срабатывает.
- Протюнить сеть
-