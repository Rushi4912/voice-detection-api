#!/usr/bin/env python3
"""
Balance training data by reducing human samples to match AI samples per language.
This prevents model bias toward the majority class.
"""
import os
import random
import shutil
from pathlib import Path

DATA_DIR = Path("training_data")
BALANCED_DIR = Path("training_data_balanced")

def balance_dataset():
    languages = ['english', 'hindi', 'malayalam', 'tamil', 'telugu']
    
    # Create balanced directory structure
    for category in ['human', 'ai']:
        for lang in languages:
            (BALANCED_DIR / category / lang).mkdir(parents=True, exist_ok=True)
    
    stats = {'before': {}, 'after': {}}
    
    for lang in languages:
        ai_dir = DATA_DIR / 'ai' / lang
        human_dir = DATA_DIR / 'human' / lang
        
        # Get file lists
        ai_files = list(ai_dir.glob('*.*')) if ai_dir.exists() else []
        human_files = list(human_dir.glob('*.*')) if human_dir.exists() else []
        
        ai_count = len(ai_files)
        human_count = len(human_files)
        
        stats['before'][lang] = {'ai': ai_count, 'human': human_count}
        
        # Sample human files to match AI count
        if human_count > ai_count:
            selected_human = random.sample(human_files, ai_count)
        else:
            selected_human = human_files
        
        # Copy AI files (all of them)
        for f in ai_files:
            shutil.copy2(f, BALANCED_DIR / 'ai' / lang / f.name)
        
        # Copy selected human files
        for f in selected_human:
            shutil.copy2(f, BALANCED_DIR / 'human' / lang / f.name)
        
        stats['after'][lang] = {'ai': ai_count, 'human': len(selected_human)}
        
        print(f"{lang.upper()}: AI={ai_count}, Human={human_count} -> Human={len(selected_human)}")
    
    # Summary
    total_before = sum(s['human'] + s['ai'] for s in stats['before'].values())
    total_after = sum(s['human'] + s['ai'] for s in stats['after'].values())
    
    print(f"\n=== SUMMARY ===")
    print(f"Total before: {total_before} files")
    print(f"Total after:  {total_after} files")
    print(f"Balanced data saved to: {BALANCED_DIR.absolute()}")
    print(f"\nTo train with balanced data:")
    print(f"./venv/bin/python3 train_models.py --data-dir training_data_balanced --epochs 20")

if __name__ == '__main__':
    balance_dataset()
