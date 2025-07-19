import json

dataset_pairs = [
    {
        "instruction": "In Troy, can I recycle pizza boxes?",
        "output": "No, pizza boxes cannot be recycled in Troy if they are greasy. The grease contaminates the paper recycling process. Please place greasy pizza boxes in the regular trash."
    },
    {
        "instruction": "What are the rules for setting out trash in NYC?",
        "output": "In New York City, trash must be set out in bins with secure lids after 6 PM the night before your collection day. Alternatively, you can place sealed bags at the curb after 8 PM."
    },
    {
        "instruction": "How can I save energy during a New York winter?",
        "output": "To save energy in winter, seal any drafts from windows and doors with weatherstripping. Lower your thermostat to 68°F when you are home and awake, and lower it further when you are asleep or away. Also, use a programmable thermostat to automate these changes."
    },
    # TODO: Add more pairs
]

output_file = "training_dataset.json"

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(dataset_pairs, f, indent=2, ensure_ascii=False)

print(f"Dataset successfully created with {len(dataset_pairs)} entries.")
print(f"File saved as: {output_file}")