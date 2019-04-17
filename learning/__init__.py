#!/usr/bin/env python3

# Declaring a method to generate new text
def sample(net, size, prime, top_k=None):
        
    if(train_on_gpu):
        net.cuda()
    else:
        net.cpu()
    
    net.eval() # eval mode
    
    # First off, run through the prime characters
    #words = [word for word in prime]
    words = []
    words.append(prime)
    h = net.init_hidden(1)
    #for word in prime:
    word, h = predict(net, words[-1], h, top_k=top_k)

    #words.append(word)
    words.append(word)
    
    # Now pass in the previous character and get a new one
    for ii in range(size): 
        word, h = predict(net, words[-1], h, top_k=top_k)
        words.append(word)

    return ''.join(words)
