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
        "subtitle_chieng.csv",
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
        
        # Special handling for subtitle_chieng.csv which has encoding issues with datasets library
        if csv_file == "subtitle_chieng.csv":
            print(f"  Using pandas fallback for {csv_file} due to encoding issues...")
            from huggingface_hub import hf_hub_download
            
            # Download the file first
            file_path = hf_hub_download(
                repo_id="MERChallenge/MER2025",
                filename=csv_file,
                repo_type="dataset"
            )
            
            # Load with pandas
            import pandas as pd
            df = pd.read_csv(file_path, encoding='utf-8')
            
            # Convert to datasets format
            from datasets import Dataset
            dataset_dict = {"train": Dataset.from_pandas(df)}
            
            # Create a clean name for the dataset
            dataset_name = csv_file.replace('.csv', '')
            datasets[dataset_name] = dataset_dict
            
            print(f"✓ Successfully loaded {csv_file} using pandas fallback")
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
        else:
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

# MERGE SUBTITLE DATA (CHINESE + ENGLISH)
print("\n" + "="*70)
print("PROCESSING SUBTITLE DATA (CHINESE + ENGLISH)")
print("="*70)

if 'subtitle_chieng' in datasets:
    subtitle_data = datasets['subtitle_chieng']['train']
    print(f"Subtitle dataset: {len(subtitle_data)} examples")
    
    # Convert to pandas DataFrame
    subtitle_df = subtitle_data.to_pandas()
    print(f"Subtitle columns: {list(subtitle_df.columns)}")
    
    print(f"✓ Processing Chinese and English subtitle columns separately")
    
    # Show sample of subtitle data
    print(f"\nSample of subtitle data (first 3 entries):")
    for i in range(min(3, len(subtitle_df))):
        row = subtitle_df.iloc[i]
        print(f"  Entry {i+1}:")
        print(f"    name: {row['name']}")
        print(f"    chinese: {row['chinese'] if pd.notna(row['chinese']) else 'N/A'}")
        print(f"    english: {row['english'] if pd.notna(row['english']) else 'N/A'}")
        print()
    
    # Create subtitle dictionary indexed by name
    subtitle_by_name = {}
    for _, row in subtitle_df.iterrows():
        name = row['name']
        subtitle_by_name[name] = {
            'name': name,
            'chinese': row['chinese'] if pd.notna(row['chinese']) else None,
            'english': row['english'] if pd.notna(row['english']) else None
        }
    
    print(f"✓ Created subtitle name-indexed dictionary with {len(subtitle_by_name)} entries")
    
    # Try to merge subtitle data with existing merged data if available
    if 'merged_by_name' in locals():
        print(f"\n🔗 INTEGRATING SUBTITLES WITH EXISTING MERGED DATA")
        print("="*60)
        
        # Find common names
        merged_names = set(merged_by_name.keys())
        subtitle_names = set(subtitle_by_name.keys())
        common_subtitle_names = merged_names.intersection(subtitle_names)
        
        print(f"- Merged dataset entries: {len(merged_names)}")
        print(f"- Subtitle dataset entries: {len(subtitle_names)}")
        print(f"- Common names with subtitles: {len(common_subtitle_names)}")
        
        # Add subtitle data to existing merged data
        enhanced_merged_by_name = {}
        for name, data in merged_by_name.items():
            enhanced_data = data.copy()
            
            # Add subtitle information if available
            if name in subtitle_by_name:
                subtitle_info = subtitle_by_name[name]
                enhanced_data.update({
                    'chinese_subtitle': subtitle_info['chinese'],
                    'english_subtitle': subtitle_info['english']
                })
            else:
                enhanced_data.update({
                    'chinese_subtitle': None,
                    'english_subtitle': None
                })
            
            enhanced_merged_by_name[name] = enhanced_data
        
        print(f"✓ Enhanced merged data with subtitle information")
        
        # Show sample of enhanced data
        print(f"\nSample of enhanced data with subtitles:")
        sample_names_with_subtitles = [name for name in list(enhanced_merged_by_name.keys())[:5] 
                                     if enhanced_merged_by_name[name]['chinese_subtitle'] or enhanced_merged_by_name[name]['english_subtitle']]
        
        for name in sample_names_with_subtitles[:2]:
            data = enhanced_merged_by_name[name]
            print(f"  Name: {name}")
            print(f"    openset: {data.get('openset', 'N/A')}")
            print(f"    reason: {data.get('reason', 'N/A')[:100]}..." if data.get('reason') and len(data.get('reason', '')) > 100 else f"    reason: {data.get('reason', 'N/A')}")
            print(f"    chinese_subtitle: {data.get('chinese_subtitle', 'N/A')}")
            print(f"    english_subtitle: {data.get('english_subtitle', 'N/A')}")
            print()
        
        # Make enhanced data globally accessible
        globals()['enhanced_merged_by_name'] = enhanced_merged_by_name
        globals()['subtitle_by_name'] = subtitle_by_name
        
        print(f"✓ Enhanced data is now available as:")
        print(f"  - 'enhanced_merged_by_name': merged data with subtitle information")
        print(f"  - 'subtitle_by_name': subtitle data indexed by name")
    
    else:
        print("ℹ️  No existing merged data found, subtitle data available separately")
        globals()['subtitle_by_name'] = subtitle_by_name
        
else:
    print("❌ Subtitle dataset not found or failed to load")
