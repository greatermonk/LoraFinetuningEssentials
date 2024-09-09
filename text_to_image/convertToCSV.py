from datasets import load_dataset
import pandas as pd


dataset = load_dataset("svjack/pokemon-blip-captions-en-zh")

for idx, image in enumerate(dataset['train'][:5]):
    print(f"Example no: {idx}\t")
    print(image)

df = pd.DataFrame(dataset["train"])
df.to_csv("pokemon-blip-captions.csv", index = False)
df.to_json('pokemon_captions.json', orient='records', lines=True)
    

