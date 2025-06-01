#!/usr/bin/env python3
import os
import sys
import shutil
import argparse

def delete_folder_contents(folder_path):
    """Delete all contents of a folder but keep the folder itself"""
    if os.path.exists(folder_path):
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)

def clean_codevector():
    delete_folder_contents("Data/CodeVector")
    print("Deleting Code vector")

def clean_processed():
    delete_folder_contents("Data/Processed")
    print("Deleting processed `.npy` audios (preemphasis, hamming window and slice)")

def clean_resultshmm():
    delete_folder_contents("Data/ResultsHMM")
    print("Deleting lambda(A,B,Pi) models")

def clean_test():
    delete_folder_contents("Data/Test")
    print("Deleting `RawDataMFCC` for Testing")

def clean_trainhmm():
    delete_folder_contents("Data/TrainHMM")
    print("Deleting `RawDataMFCC` for Training HMM")

def clean_all():
    response = input("Are you sure you want to delete all data? (y/n): ")
    if response.lower() == 'y':
        clean_codevector()
        clean_processed()
        clean_resultshmm()
        clean_test()
        clean_trainhmm()
        print("All data cleaned successfully")
    else:
        print("Operation cancelled")

def clean_all_except(except_option):
    response = input(f"Are you sure you want to delete all data except {except_option}? (y/n): ")
    if response.lower() == 'y':
        all_options = ['codevector', 'processed', 'resultshmm', 'test', 'trainhmm']
        cleanup_functions = {
            'codevector': clean_codevector,
            'processed': clean_processed,
            'resultshmm': clean_resultshmm,
            'test': clean_test,
            'trainhmm': clean_trainhmm
        }
        
        for option in all_options:
            if option != except_option:
                cleanup_functions[option]()
        
        print(f"All data cleaned successfully (except {except_option})")
    else:
        print("Operation cancelled")

def main():
    parser = argparse.ArgumentParser(description='Clean data directories')
    parser.add_argument('option', nargs='?', choices=['codevector', 'processed', 'resultshmm', 'test', 'trainhmm', 'all'],
                        help='Choose which data to clean')
    parser.add_argument('--except', '-e', dest='except_option', 
                        choices=['codevector', 'processed', 'resultshmm', 'test', 'trainhmm'],
                        help='When using "all", exclude this option from cleaning')
    
    # Handle case with no arguments to show help
    if len(sys.argv) == 1:
        print("Available options:")
        print("  codevector  - Delete Code vector")
        print("  processed   - Delete processed `.npy` audios (preemphasis, hamming window and slice)")
        print("  resultshmm  - Delete lambda(A,B,Pi) models")
        print("  test        - Delete `RawDataMFCC` for Testing")
        print("  trainhmm    - Delete `RawDataMFCC` for Training HMM")
        print("  all         - Delete all of the above (with confirmation)")
        print("\nUsage: python clean_data.py <option>")
        print("       python clean_data.py all --except <option>")
        sys.exit(1)
    
    args = parser.parse_args()
    
    # Handle --except flag without 'all' option
    if args.except_option and args.option != 'all':
        print("Error: --except can only be used with 'all' option")
        sys.exit(1)
    
    options = {
        'codevector': clean_codevector,
        'processed': clean_processed,
        'resultshmm': clean_resultshmm,
        'test': clean_test,
        'trainhmm': clean_trainhmm,
        'all': lambda: clean_all_except(args.except_option) if args.except_option else clean_all()
    }
    
    if args.option in options:
        options[args.option]()
    else:
        print(f"Invalid option: {args.option}")
        sys.exit(1)

if __name__ == "__main__":
    main()
