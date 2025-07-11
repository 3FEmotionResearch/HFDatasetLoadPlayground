from datasets import load_dataset
from huggingface_hub import list_repo_files
import pandas as pd

# Discover files using Hugging Face Hub API
print("DISCOVERING FILES IN THE DATASET:")
print("="*50)

try:
    # List all files in the dataset repository
    files = list_repo_files("MERChallenge/MER2025", repo_type="dataset")
    
    print("All files in the dataset:")
    for file in files:
        print(f"  - {file}")
    
    # Filter for CSV files
    csv_files = [f for f in files if f.endswith('.csv')]
    print(f"\nFound {len(csv_files)} CSV files:")
    for csv_file in csv_files:
        print(f"  - {csv_file}")
    
except Exception as e:
    print(f"Error listing files: {e}")
    print("Falling back to manual file list...")
    
    # Fallback to the files we saw in the download output
    csv_files = [
        "track1_train_disdim.csv",
        "track2_train_mercaptionplus.csv", 
        "track2_train_ovmerd.csv",
        "track3_train_mercaptionplus.csv",
        "track3_train_ovmerd.csv"
    ]
    print(f"Using files from download output: {csv_files}")

print("\n" + "="*50)
print("LOADING INDIVIDUAL CSV FILES:")
print("="*50)

datasets = {}
for csv_file in csv_files:
    try:
        print(f"\nLoading {csv_file}...")
        
        # Load without specifying split to see all available splits
        dataset_dict = load_dataset(
            "MERChallenge/MER2025", 
            data_files=csv_file
        )
        
        # Create a clean name for the dataset
        dataset_name = csv_file.replace('.csv', '')
        datasets[dataset_name] = dataset_dict
        
        print(f"✓ Successfully loaded {csv_file}")
        print(f"  Available splits: {list(dataset_dict.keys())}")
        
        # Show details for each split
        for split_name, split_data in dataset_dict.items():
            print(f"  {split_name} split:")
            print(f"    - Examples: {len(split_data)}")
            print(f"    - Features: {split_data.features}")
            print(f"    - Column names: {list(split_data.features.keys())}")
            
            # Show first example if available
            if len(split_data) > 0:
                print(f"    - First example keys: {list(split_data[0].keys())}")
                print(f"    - First example: {split_data[0]}")
            
    except Exception as e:
        print(f"✗ Failed to load {csv_file}: {e}")

print(f"\nSuccessfully loaded {len(datasets)} datasets")

# Show summary of all loaded datasets and their splits
if datasets:
    print("\n" + "="*50)
    print("SUMMARY OF ALL DATASETS AND SPLITS:")
    print("="*50)
    for dataset_name, dataset_dict in datasets.items():
        print(f"\n{dataset_name}:")
        for split_name, split_data in dataset_dict.items():
            print(f"  {split_name} split:")
            print(f"    - Examples: {len(split_data)}")
            print(f"    - Columns: {list(split_data.features.keys())}")
            print(f"    - Column types: {split_data.features}")

# MERGE TRACK2 AND TRACK3 MERCAPTIONPLUS DATASETS
print("\n" + "="*70)
print("MERGING track2_train_mercaptionplus AND track3_train_mercaptionplus")
print("="*70)

# Check if we have both datasets
if ('track2_train_mercaptionplus' in datasets and 
    'track3_train_mercaptionplus' in datasets):
    
    # Get the train splits from both datasets
    track2_data = datasets['track2_train_mercaptionplus']['train']
    track3_data = datasets['track3_train_mercaptionplus']['train']
    
    print(f"Track2 dataset: {len(track2_data)} examples")
    print(f"Track3 dataset: {len(track3_data)} examples")
    
    # Convert to pandas DataFrames for easier merging
    track2_df = track2_data.to_pandas()
    track3_df = track3_data.to_pandas()
    
    print(f"\nTrack2 columns: {list(track2_df.columns)}")
    print(f"Track3 columns: {list(track3_df.columns)}")
    
    # Check if both have 'name' column
    if 'name' in track2_df.columns and 'name' in track3_df.columns:
        print(f"\nBoth datasets have 'name' column. Proceeding with merge...")
        
        # Merge on 'name' column
        merged_df = pd.merge(
            track2_df, 
            track3_df, 
            on='name', 
            how='inner',  # Only keep rows that exist in both datasets
            suffixes=('_track2', '_track3')
        )
        
        print(f"✓ Merged dataset created with {len(merged_df)} rows")
        print(f"Merged columns: {list(merged_df.columns)}")
        
        # Create a list of dictionaries with combined data
        merged_data = []
        for _, row in merged_df.iterrows():
            combined_row = {
                'name': row['name'],
                'openset': row.get('openset', None),  # from track2
                'reason': row.get('reason', None)     # from track3
            }
            
            # Add all other columns from both datasets
            for col in merged_df.columns:
                if col not in ['name', 'openset', 'reason']:
                    combined_row[col] = row[col]
            
            merged_data.append(combined_row)
        
        print(f"\n✓ Created merged data structure with {len(merged_data)} entries")
        
        # Show sample of merged data
        print(f"\nSample of merged data (first 3 entries):")
        for i, entry in enumerate(merged_data[:3]):
            print(f"  Entry {i+1}:")
            print(f"    name: {entry['name']}")
            print(f"    openset: {entry.get('openset', 'N/A')}")
            print(f"    reason: {entry.get('reason', 'N/A')}")
            print(f"    all keys: {list(entry.keys())}")
            print()
        
        # Create a dictionary indexed by name for easy access
        merged_by_name = {}
        for entry in merged_data:
            name = entry['name']
            merged_by_name[name] = entry
        
        print(f"✓ Created name-indexed dictionary with {len(merged_by_name)} entries")
        
        # Example of accessing data
        print(f"\nExample of accessing merged data:")
        sample_names = list(merged_by_name.keys())[:2]
        for name in sample_names:
            data = merged_by_name[name]
            print(f"  Name: {name}")
            print(f"    openset: {data.get('openset', 'N/A')}")
            print(f"    reason: {data.get('reason', 'N/A')}")
            print()
    else:
        print("❌ Cannot merge: One or both datasets don't have 'name' column")
else:
    print("❌ Cannot merge: Required datasets not found")
    if 'track2_train_mercaptionplus' not in datasets:
        print("  - track2_train_mercaptionplus not loaded")
    if 'track3_train_mercaptionplus' not in datasets:
        print("  - track3_train_mercaptionplus not loaded")
