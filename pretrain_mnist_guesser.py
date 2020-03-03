# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 12:23:11 2020

@author: urixs
"""
import numpy as np
from sklearn.model_selection import train_test_split
from itertools import count
from sklearn.metrics import confusion_matrix
import argparse
from collections import deque
import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--save_dir",
                    type=str,
                    default='./pretrained_mnist_guesser_models',
                    help="Directory for saved models")
parser.add_argument("--hidden-dim",
                    type=int,
                    default=256,
                    help="Hidden dimension")
parser.add_argument("--lr",
                    type=float,
                    default=1e-4,
                    help="Learning rate")
parser.add_argument("--weight_decay",
                    type=float,
                    default=0.,
                    help="l_2 weight penalty")
parser.add_argument("--case",
                    type=int,
                    default=2,
                    help="Which data to use")
parser.add_argument("--val_trials_wo_im",
                    type=int,
                    default=50,
                    help="Number of validation trials without improvement")
parser.add_argument("--val_interval",
                    type=int,
                    default=1000,
                    help="Interval for calculating validation reward and saving model")

FLAGS = parser.parse_args(args=[])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class Guesser(nn.Module):
    """
    implements a net that guesses the outcome given the state
    """
    
    def __init__(self, 
                 state_dim, 
                 hidden_dim=FLAGS.hidden_dim, 
                 num_classes=10):
        super(Guesser, self).__init__()
        
        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.PReLU(), 
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.PReLU(),
        )
        
        self.layer3 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.PReLU(),
        )
        
        
        # output layer
        self.logits = nn.Linear(hidden_dim, num_classes)
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.optimizer = torch.optim.Adam(self.parameters(), 
                                          weight_decay=FLAGS.weight_decay,
                                          lr=FLAGS.lr)


    def forward(self, x):
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        logits = self.logits(x)
        probs = F.softmax(logits, dim=1)
        
        return logits, probs   
    
    def _to_variable(self, x: np.ndarray) -> torch.Tensor:
        """torch.Variable syntax helper
        Args:
            x (np.ndarray): 2-D tensor of shape (n, input_dim)
        Returns:
            torch.Tensor: torch variable
        """
        return torch.autograd.Variable(torch.Tensor(x))
    
def save_network(i_episode, acc=None):
    """ A function that saves the gesser params"""
    
    if not os.path.exists(FLAGS.save_dir):
        os.makedirs(FLAGS.save_dir)
    
    if i_episode == 'best':
        guesser_filename = 'best_guesser.pth'
    else:
        guesser_filename = '{}_{}_{:1.3f}.pth'.format(i_episode, 'guesser', acc)
        
    guesser_save_path = os.path.join(FLAGS.save_dir, guesser_filename)
    
    # save guesser
    if os.path.exists(guesser_save_path):
        os.remove(guesser_save_path)
    torch.save(guesser.cpu().state_dict(), guesser_save_path + '~')
    guesser.to(device=device)
    os.rename(guesser_save_path + '~', guesser_save_path)

# Load data and randomly split to train, validation and test sets
n_questions = 28 * 28

# Initialize guesser
guesser = Guesser(2 * n_questions)  
guesser.to(device=device)      
       

X_train, X_test, y_train, y_test = utils.load_mnist(case=FLAGS.case)
 
X_train, X_val, y_train, y_val = train_test_split(X_train, 
                                                  y_train, 
                                                  test_size=0.33)


if len(X_val > 10000):
    X_val = X_val[:10000]
    y_val = y_val[:10000]
        
def main():
    """ Main """
    
    # Delete models from earlier runs
    if os.path.exists(FLAGS.save_dir):
        shutil.rmtree(FLAGS.save_dir)
    
    # Reset counter
    val_trials_without_improvement = 0
    
    
    losses = deque(maxlen=100)
    best_val_acc = 0
    
    for i in count(1):
     patient = np.random.randint(X_train.shape[0])
     x = X_train[patient]
     x = np.concatenate([x, np.ones(n_questions)])
     guesser_input = guesser._to_variable(x.reshape(-1, 2 * n_questions))
     guesser_input = guesser_input.to(device=device)
     guesser.train(mode=False)
     logits, probs = guesser(guesser_input)
     y_true = y_train[patient]
     y = torch.Tensor([y_true]).long()
     y = y.to(device=device)
     guesser.optimizer.zero_grad()             
     guesser.train(mode=True)
     loss = guesser.criterion(logits, y) 
     losses.append(loss.item())       
     loss.backward()
     guesser.optimizer.step()
     
     if i % 100 == 0:
         print('Step: {}, loss={:1.3f}'.format(i, loss.item()))
     
        # COmpute performance on validation set and reset counter if necessary    
     if i % FLAGS.val_interval == 0:
        new_best_val_acc = val(i_episode=i, best_val_acc=best_val_acc)
        if new_best_val_acc > best_val_acc:
                    best_val_acc = new_best_val_acc
                    val_trials_without_improvement = 0
        else:
            val_trials_without_improvement += 1
            
    # check whether to stop training
        if val_trials_without_improvement == FLAGS.val_trials_wo_im:
            print('Did not achieve val acc improvement for {} trials, training is done.'.format(FLAGS.val_trials_wo_im))
            break


def val(i_episode : int, 
        best_val_acc : float) -> float:
    """ Computes performance on validation set """
    
    print('Running validation')
    y_hat_val = np.zeros(len(y_val))
    
    for i in range(len(X_val)):
        x = X_val[i]
        x = np.concatenate([x, np.ones(n_questions)])
        guesser_input = guesser._to_variable(x.reshape(-1, 2 * n_questions))
        guesser_input = guesser_input.to(device=device)
        guesser.train(mode=False)
        logits, probs = guesser(guesser_input)
        y_hat_val[i] = torch.argmax(probs).item()

    confmat = confusion_matrix(y_val,  y_hat_val)
    acc = np.sum(np.diag(confmat)) / len(y_val)
    #save_network(i_episode, acc)
    
    if acc > best_val_acc:
        print('New best Acc acheievd, saving best model')
        save_network(i_episode='best')
        save_network(i_episode, acc)
        
        return acc
    
    else:
        return best_val_acc
    

if __name__ == '__main__':
    main()