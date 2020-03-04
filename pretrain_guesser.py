# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 12:23:11 2020

@author: urixs
"""
import numpy as np
from sklearn.model_selection import train_test_split
from itertools import count
from sklearn.metrics import roc_auc_score
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
                    default='./pretrained_guesser_models',
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
                    default=122,
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
                 num_classes=2):
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
    
def save_network(i_episode, auc=None):
    """ A function that saves the gesser params"""
    
    if not os.path.exists(FLAGS.save_dir):
        os.makedirs(FLAGS.save_dir)
    
    if i_episode == 'best':
        guesser_filename = 'best_guesser.pth'
    else:
        guesser_filename = '{}_{}_{:1.3f}.pth'.format(i_episode, 'guesser', auc)
        
    guesser_save_path = os.path.join(FLAGS.save_dir, guesser_filename)
    
    # save guesser
    if os.path.exists(guesser_save_path):
        os.remove(guesser_save_path)
    torch.save(guesser.cpu().state_dict(), guesser_save_path + '~')
    guesser.to(device=device)
    os.rename(guesser_save_path + '~', guesser_save_path)

# Load data and randomly split to train, validation and test sets
X, y, question_names, class_names, scaler  = utils.load_data(case=FLAGS.case)
n_questions = X.shape[1]

# Initialize guesser
guesser = Guesser(2 * n_questions)   
guesser.to(device=device)      

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                     y, 
                                                     test_size=0.33, 
                                                     random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, 
                                                  y_train, 
                                                  test_size=0.05, 
                                                  random_state=24)

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
    
    # Store indices of positive and negative patients for oversampling
    class_0_inds = [index for index,value in enumerate(y_train) if value == 0]
    class_1_inds = [index for index,value in enumerate(y_train) if value == 1] 


    
    losses = deque(maxlen=100)
    best_val_auc = 0
    
    for i in count(1):
     #patient = np.random.randint(X_train.shape[0])
     if np.random.rand() < .5:
         ind = np.random.randint(len(class_0_inds))
         patient = class_0_inds[ind]
     else:
         ind = np.random.randint(len(class_1_inds))
         patient = class_1_inds[ind]
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
        new_best_val_auc = val(i_episode=i, best_val_auc=best_val_auc)
        if new_best_val_auc > best_val_auc:
                    best_val_auc = new_best_val_auc
                    val_trials_without_improvement = 0
        else:
            val_trials_without_improvement += 1
            
    # check whether to stop training
        if val_trials_without_improvement == FLAGS.val_trials_wo_im:
            print('Did not achieve val AUC improvement for {} trials, training is done.'.format(FLAGS.val_trials_wo_im))
            break


def val(i_episode : int, 
        best_val_auc : float) -> float:
    """ Computes performance on validation set """
    
    print('Running validation')
    y_hat_val_prob = np.zeros(len(y_val))
    
    for i in range(len(X_val)):
        x = X_val[i]
        x = np.concatenate([x, np.ones(n_questions)])
        guesser_input = guesser._to_variable(x.reshape(-1, 2 * n_questions))
        guesser_input = guesser_input.to(device=device)
        guesser.train(mode=False)
        logits, probs = guesser(guesser_input)
        y_hat_val_prob[i] = probs.squeeze()[1].item()

    roc_auc_score_ = roc_auc_score(y_val,  y_hat_val_prob)
    
    
    if roc_auc_score_ > best_val_auc:
        print('New best AUC acheievd, saving best model')
        save_network(i_episode='best')
        save_network(i_episode, roc_auc_score_)
        
        return roc_auc_score_
    
    else:
        return best_val_auc
    

if __name__ == '__main__':
    main()