#!/usr/bin/env python3
"""
AI-News/learning/vectorization.py 
Modified from GitHub repository source: authored by Igor Dzreyev from https://github.com/joosthub/PyTorchNLPBook
Attribution: "Natural Language Processing with PyTorch by Delip Rao and Brian McMahan (O'Reilly). Copyright 2019, Delip Rao and Brian McMahan, 978-1-491-97823-8"

The source code is rightfully reproduced under the Apache 2.0 License for academic research at Baldwin-Wallace University, Dept. of Computer Science.
"""
from argparse import Namespace
import os
import torch

def utils(frequency_cutoff=25, 
			model_state_file='model.pth', 
			review_csv='data/reviews.csv', 
			save_dir='../model/',
			vectorizer_file='vectorizer.json', 
			batch_size=128, 
			early_stopping_criteria=5,
			learning_rate=0.001, 
			num_epochs=100, 
			seed=1110, 
			catch_keyboard_interrupt=True,
			expand_filepaths_to_save_dir=True,
			reload_from_files=False):
    args = Namespace(
        # Data and Path information
        frequency_cutoff=frequency_cutoff,
        model_state_file=model_state_file,
        review_csv=review_csv,
        save_dir=save_dir,
        vectorizer_file='vectorizer.json',
        # No Model hyper parameters <-- Need these
        # Training hyper parameters <-- Need these
        batch_size=batch_size,
        early_stopping_criteria=early_stopping_criteria,
        # Investigate learning rate for interaction with ./optimization.py
        # And its adaptive learning rate algorithim
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        seed=seed,
        # Runtime options
        catch_keyboard_interrupt=catch_keyboard_interrupt,
        cuda=True,
        expand_filepaths_to_save_dir=True,
        reload_from_files=reload_from_files,
    )

	if args.expand_filepaths_to_save_dir:
	    args.vectorizer_file = os.path.join(args.save_dir,
	                                        args.vectorizer_file)

	    args.model_state_file = os.path.join(args.save_dir,
	                                         args.model_state_file)
	    
	    print("Expanded filepaths: ")
	    print("\t{}".format(args.vectorizer_file))
	    print("\t{}".format(args.model_state_file))
	    
	# Check CUDA
	if not torch.cuda.is_available():
	    args.cuda = False

	print("Using CUDA: {}".format(args.cuda))

	args.device = torch.device("cuda" if args.cuda else "cpu")

	# Set seed for reproducibility
	set_seed_everywhere(args.seed, args.cuda)

	# handle dirs
	handle_dirs(args.save_dir)

	return args