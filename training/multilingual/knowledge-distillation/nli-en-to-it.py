import datasets

ted = datasets.load_dataset('ted_multi', split='train')
ted

from tqdm.auto import tqdm  # so we see progress bar

print(f"Before: {len(ted)}")

# create dict to store our pairs
train_samples = {f'en-it': []}

# now build our training samples list
for row in tqdm(ted):
    # get source (English)
    idx = row['translations']['language'].index('en')
    source = row['translations']['translation'][idx].strip()
    # get target (Italian)
    try:
        idx = row['translations']['language'].index('it')
        target = row['translations']['translation'][idx].strip()
    except ValueError:
        continue
    # append to training examples
    train_samples[f'en-it'].append(
        source+'\t'+target
    )

print(f"After: {len(train_samples)}")

import gzip

# save to file, sentence transformers reader will expect tsv.gz file
for lang_pair in train_samples.keys():
    with gzip.open('ted-train-en-it.tsv.gz', 'wt', encoding='utf-8') as f:
        f.write('\n'.join(train_samples[lang_pair]))
        
mnli = datasets.load_dataset(
    "MoritzLaurer/multilingual-NLI-26lang-2mil7",
    split="it_mnli[:5000]"
)

import numpy as np

np.random.seed(0)  # for reproducibility
negative_size = 32  # higher number makes it harder

it_texts = mnli['hypothesis']

it_eval = []

mnli = mnli.filter(lambda x: x['label'] == 0)

for row in tqdm(mnli):
    anchor = row['premise']
    positive = row['hypothesis']
    # get random set of negative samples
    sample = np.random.choice(
        it_texts,
        negative_size,
        replace=False
    )
    it_eval.append({
        'query': anchor,
        'positive': positive,
        'negative': sample.tolist()
    })

print(f"it_eval: {len(it_eval)}")

en_texts = mnli['hypothesis_original']
en_eval = []

mnli = mnli.filter(lambda x: x['label'] == 0)

for row in tqdm(mnli):
    anchor = row['premise_original']
    positive = row['hypothesis_original']
    # get random set of negative samples
    sample = np.random.choice(
        en_texts,
        negative_size,
        replace=False
    )
    en_eval.append({
        'query': anchor,
        'positive': positive,
        'negative': sample.tolist()
    })
    
print(f"en_eval: {len(en_eval)}")

# we would expect en and it evaluation sets to be equal
assert len(en_eval) == len(it_eval)

from sentence_transformers.evaluation import RerankingEvaluator

en_evaluator = RerankingEvaluator(en_eval)
it_evaluator = RerankingEvaluator(it_eval)

from sentence_transformers import models, SentenceTransformer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

xlmr = models.Transformer('xlm-roberta-base')
pooler = models.Pooling(
    xlmr.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True
)

student = SentenceTransformer(
    modules=[xlmr, pooler],
    device=device+":0"
)

print("student perf on evaluators:")
print(f"EN: {en_evaluator(student)}")
print(f"IT: {it_evaluator(student)}")

from sentence_transformers import SentenceTransformer

teacher = SentenceTransformer(
    'jamescalam/mpnet-snli-negatives',
    device=device+":1"
)

print("teacher perf on evaluators:")
print(f"EN: {en_evaluator(teacher)}")
print(f"IT: {it_evaluator(teacher)}")

from sentence_transformers import ParallelSentencesDataset

data = ParallelSentencesDataset(
    student_model=student,
    teacher_model=teacher,
    batch_size=48,
    use_embedding_cache=True
)

data.load_data(
    'ted-train-en-it.tsv.gz',
    max_sentence_length=500
)

from torch.utils.data import DataLoader

loader = DataLoader(
    data,
    shuffle=True,
    batch_size=48
)

from sentence_transformers import losses

loss = losses.MSELoss(model=student)

from sentence_transformers import evaluation
import numpy as np

epochs = 5
warmup_steps = int(len(loader) * epochs * 0.1)

student.fit(
    train_objectives=[(loader, loss)],
    epochs=epochs,
    warmup_steps=warmup_steps,
    output_path='xlmr-roberta-en-it',
    optimizer_params={'lr': 2e-5},
    save_best_model=True,
    show_progress_bar=True,
    evaluator=it_evaluator,
    evaluation_steps=100  # every 4800 samples
)