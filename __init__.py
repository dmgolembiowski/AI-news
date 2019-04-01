#!/usr/bin/env python3

def HELP():
    message = """
NAME
    AI-News - artificially intelligent fake-news articles


SYNOPSIS
    python AI-News [MODE] [OPTION]...

DESCRIPTION
    AI-News is a program in Python 3.7 that uses PyTorch to train a Gated Recurrent Unit (GRU)
    neural network to write fake news articles. The network was trained using supervised
    learning via SQLite training data at https://www.dropbox.com/s/b2cyb85ib17s7zo/all-the-news.db?dl=0%22 . 
MODE
    --train, -t
        Launch a supervised training session using nvidia GPU hardware
    
    --serve, -fs
        Launch a flask server to http://127.0.0.1:5000 to see news article output

    --no-serve, -x
        Do not launch a flask server to see news article output

OPTION        
    --save, -s [location]
        Save program output to file

    --print, -p
        Print program output to the console

ACKNOWLEDGMENTS
    This project would not be possible without the support of 
    + Sydney Leither & Jeffery <last name>;
    + Chris Midkiff, Ian Walton, & Josh Neubecker; 
    + Delip Rao & Bian McMahan;
    + John Paximadis & Andrew Watkins;
    and so many others who supported our academic careers and intellectual endeavors.
    THANK YOU!
"""

def main():
    """

    """
    from gate import Update, Reset
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

