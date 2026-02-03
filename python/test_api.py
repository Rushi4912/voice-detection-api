#!/usr/bin/env python3
"""
Test script to validate the AI Voice Detection API with sample audio files.
This script tests both Human and AI audio samples from the training data.

Usage:
    python test_api.py                    # Test with default samples
    python test_api.py --samples 10       # Test with 10 samples per category
    python test_api.py --url http://localhost:3000  # Test against Node.js API
"""

import requests
import base64
import json
import random
import argparse
from pathlib import Path
from typing import List, Tuple
import sys

# Configuration
PYTHON_API_URL = "http://localhost:5000/analyze"
NODEJS_API_URL = "http://localhost:3000/api/detect"
TRAINING_DATA_DIR = Path(__file__).parent / "training_data_balanced"

def load_audio_file(file_path: Path) -> str:
    """Load audio file and convert to base64"""
    with open(file_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def get_sample_files(category: str, language: str, num_samples: int = 5) -> List[Path]:
    """Get random sample files from a category/language"""
    folder = TRAINING_DATA_DIR / category / language
    if not folder.exists():
        return []
    
    files = list(folder.glob("*.mp3")) + list(folder.glob("*.wav"))
    if len(files) > num_samples:
        return random.sample(files, num_samples)
    return files

def test_single_file(api_url: str, file_path: Path, expected_label: str, language: str) -> dict:
    """Test a single audio file against the API"""
    try:
        audio_base64 = load_audio_file(file_path)
        
        payload = {
            "audio_base64": audio_base64,
            "language": language,
        }
        
        response = requests.post(api_url, json=payload, timeout=60)
        result = response.json()
        
        # Determine if prediction matches expected
        if 'classification' in result:
            predicted = result['classification']
        elif 'result' in result and 'classification' in result['result']:
            predicted = result['result']['classification']
        else:
            predicted = "UNKNOWN"
        
        correct = (predicted == expected_label)
        
        return {
            'file': file_path.name,
            'expected': expected_label,
            'predicted': predicted,
            'correct': correct,
            'confidence': result.get('confidence', result.get('result', {}).get('confidence', 'N/A')),
            'response': result
        }
        
    except Exception as e:
        return {
            'file': file_path.name,
            'expected': expected_label,
            'predicted': 'ERROR',
            'correct': False,
            'error': str(e)
        }

def run_tests(api_url: str, num_samples: int = 5) -> dict:
    """Run tests on sample files and report results"""
    print(f"\n{'='*60}")
    print(f"🧪 AI Voice Detection API Test")
    print(f"{'='*60}")
    print(f"API URL: {api_url}")
    print(f"Samples per category: {num_samples}")
    print(f"{'='*60}\n")
    
    languages = ['english', 'hindi', 'malayalam', 'tamil', 'telugu']
    results = {
        'total': 0,
        'correct': 0,
        'human_correct': 0,
        'human_total': 0,
        'ai_correct': 0,
        'ai_total': 0,
        'details': []
    }
    
    for language in languages:
        print(f"\n📁 Testing {language.upper()}...")
        
        # Test Human samples
        human_files = get_sample_files('human', language, num_samples)
        for file_path in human_files:
            result = test_single_file(api_url, file_path, 'HUMAN', language.capitalize())
            results['details'].append(result)
            results['total'] += 1
            results['human_total'] += 1
            if result['correct']:
                results['correct'] += 1
                results['human_correct'] += 1
            
            status = "✅" if result['correct'] else "❌"
            print(f"  {status} [HUMAN] {result['file'][:30]:30} → {result['predicted']} (conf: {result.get('confidence', 'N/A')})")
        
        # Test AI samples
        ai_files = get_sample_files('ai', language, num_samples)
        for file_path in ai_files:
            result = test_single_file(api_url, file_path, 'AI_GENERATED', language.capitalize())
            results['details'].append(result)
            results['total'] += 1
            results['ai_total'] += 1
            if result['correct']:
                results['correct'] += 1
                results['ai_correct'] += 1
            
            status = "✅" if result['correct'] else "❌"
            print(f"  {status} [AI]    {result['file'][:30]:30} → {result['predicted']} (conf: {result.get('confidence', 'N/A')})")
    
    # Summary
    accuracy = (results['correct'] / results['total'] * 100) if results['total'] > 0 else 0
    human_accuracy = (results['human_correct'] / results['human_total'] * 100) if results['human_total'] > 0 else 0
    ai_accuracy = (results['ai_correct'] / results['ai_total'] * 100) if results['ai_total'] > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"📊 TEST RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Total Samples:     {results['total']}")
    print(f"  Correct:           {results['correct']}")
    print(f"  Overall Accuracy:  {accuracy:.2f}%")
    print(f"  ")
    print(f"  Human Detection:   {results['human_correct']}/{results['human_total']} ({human_accuracy:.2f}%)")
    print(f"  AI Detection:      {results['ai_correct']}/{results['ai_total']} ({ai_accuracy:.2f}%)")
    print(f"{'='*60}\n")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Test AI Voice Detection API')
    parser.add_argument('--samples', type=int, default=3, help='Number of samples per category/language')
    parser.add_argument('--url', type=str, default=PYTHON_API_URL, help='API URL to test')
    parser.add_argument('--nodejs', action='store_true', help='Test Node.js API instead of Python')
    
    args = parser.parse_args()
    
    api_url = NODEJS_API_URL if args.nodejs else args.url
    
    # Check if API is reachable
    print(f"🔍 Checking API connection to {api_url}...")
    try:
        # Try health endpoint first
        health_url = api_url.rsplit('/', 1)[0] + '/health'
        response = requests.get(health_url, timeout=5)
        print(f"✅ API is reachable (status: {response.status_code})")
    except:
        print(f"⚠️  Could not reach health endpoint, proceeding with tests...")
    
    # Run tests
    results = run_tests(api_url, args.samples)
    
    # Exit with error code if accuracy is below threshold
    accuracy = (results['correct'] / results['total'] * 100) if results['total'] > 0 else 0
    if accuracy < 70:
        print("⚠️  Warning: Accuracy is below 70%!")
        sys.exit(1)
    else:
        print("✅ All tests passed!")
        sys.exit(0)

if __name__ == '__main__':
    main()
