# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 21:49:57 2019

@author: urixs

Environment for MNIST

"""
import argparse
import numpy as np
import os
from sklearn.model_selection import train_test_split
import gym
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.nn.functional as F

import utils


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
parser.add_argument("--min_lr",
                    type=float,
                    default=1e-6,
                    help="Minimal learning rate")
parser.add_argument("--decay_step_size",
                    type=int,
                    default=12500,
                    help="LR decay step size")
parser.add_argument("--lr_decay_factor",
                    type=float,
                    default=0.1,
                    help="LR decay factor")


FLAGS = parser.parse_args()

def lambda_rule(i_episode) -> float:
    """ stepwise learning rate calculator """
    exponent = int(np.floor((i_episode + 1) / FLAGS.decay_step_size))
    return np.power(FLAGS.lr_decay_factor, exponent)


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
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, 
                                               lr_lambda=lambda_rule)


    def forward(self, x):
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        logits = self.logits(x)
        probs = F.softmax(logits, dim=1)
        
        return logits, probs   
    
    def update_learning_rate(self):
        """ Learning rate updater """
        
        self.scheduler.step()
        lr = self.optimizer.param_groups[0]['lr']
        if lr < FLAGS.min_lr:
            self.optimizer.param_groups[0]['lr'] = FLAGS.min_lr
            lr = self.optimizer.param_groups[0]['lr']
        print('Guesser learning rate = %.7f' % lr)
    
    def _to_variable(self, x: np.ndarray) -> torch.Tensor:
        """torch.Variable syntax helper
        Args:
            x (np.ndarray): 2-D tensor of shape (n, input_dim)
        Returns:
            torch.Tensor: torch variable
        """
        return torch.autograd.Variable(torch.Tensor(x))


class Mnist_env(gym.Env):
     """ Questionnaire Environment class
        Args:
            case (int): which data to use
            oversample (Boolean): whether to oversample the small class
            load_pretrained_guesser (Boolean): whether to load a pretrained guesser
        """
        
     def __init__(self, 
                  case=1, 
                  oversample=True,
                  load_pretrained_guesser=True,
                  episode_length=5):
         
         # Load data
         self.n_questions = 28 * 28
         self.X_train, self.X_test, self.y_train, self.y_test = utils.load_mnist(case=case)
         
         self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, 
                                                                               self.y_train, 
                                                                               test_size=0.008)
         
         self.guesser = Guesser(self.n_questions)
         
         self.episode_length = episode_length
         
         # Load pre-trained guesser network, if needed
         if load_pretrained_guesser:
             save_dir = './mnist_pretrained_guesser_models'
             guesser_filename = 'best_guesser.pth'
             guesser_load_path = os.path.join(save_dir, guesser_filename)
             if os.path.exists(guesser_load_path):
                 print('Loading pre-trained guesser')
                 guesser_state_dict = torch.load(guesser_load_path)
                 self.guesser.load_state_dict(guesser_state_dict)
         
         print('Initialized questionnaire environment')                  
     
    # Reset environment
     def reset(self, 
               mode='training', 
               patient=0,
               train_guesser=True):
         """ 
         Args: mode: training / val / test
               patient (int): index of patient
               train_guesser (Boolean): flag indicating whether to train guesser network in this episode
         
         Selects a patient (random for training, or pre-defined for val and test) ,
         Resets the state to contain the basic information,
         Resets 'done' flag to false,
         Resets 'train_guesser' flag
         """
         
         self.state =  np.zeros(self.n_questions)
        
         if  mode == 'training':
             self.patient = np.random.randint(self.X_train.shape[0])              
         else: 
             self.patient = patient
             
         self.done = False
         self.s = np.array(self.state)
         self.time = 0
         if mode == 'training':
             self.train_guesser = train_guesser
         else:
             self.train_guesser = False
         return self.s
        
     def reset_mask(self):
         """ A method that resets the mask that is applied 
         to the q values, so that questions that were already 
         asked will not be asked again.
         """
         mask = torch.ones(self.n_questions + 1)
         
         return mask
                     
     def step(self, 
              action, 
              mode='training'):
         """ State update mechanism """
         
         # update state
         next_state = self.update_state(action, mode)
         self.state = np.array(next_state)     
         self.s = np.array(self.state)
                   
         # compute reward
         self.reward = self.compute_reward(mode)
         
         self.time += 1
         if self.time == self.episode_length:
             self.terminate_episode()
        
         return self.s, self.reward, self.done, self.guess
     
     # Update 'done' flag when episode terminates   
     def terminate_episode(self):
         self.done = True
            
     def update_state(self, action, mode):
         next_state = np.array(self.state)
         
         if action < self.n_questions: # Not making a guess
             if  mode == 'training':
                 next_state[action] = self.X_train[self.patient, action]        
             elif mode == 'val':
                 next_state[action] = self.X_val[self.patient, action]
             elif mode == 'test': 
                 next_state[action] = self.X_test[self.patient, action]           
             self.guess = -1
             self.done = False
             
         else: # Making a guess
              # run guesser, and store guess and outcome probability
              guesser_input = self.guesser._to_variable(self.state.reshape(-1, self.n_questions))
              self.guesser.train(mode=False)
              self.logits, self.probs = self.guesser(guesser_input)
              self.guess = np.argmax(self.probs.detach().numpy().squeeze())
              self.outcome_prob = self.probs.detach().numpy().squeeze()[1]
              if mode == 'training':
                  # store probability of true outcome for reward calculation
                  self.correct_prob = self.probs.detach().numpy().squeeze()[self.y_train[self.patient]]
              self.terminate_episode()
             
         return next_state
            
     def compute_reward(self, mode):
         """ Compute the reward """
         
         if mode == 'test':
             return None
         if mode == 'training':
             y_true = self.y_train[self.patient]
         
         if self.guess == -1: # no guess was made
             return .01 * np.random.rand()
         else:
             reward = self.correct_prob
         if self.train_guesser:
             # train guesser
             self.guesser.optimizer.zero_grad()             
             y = torch.Tensor([y_true]).long()
             self.guesser.train(mode=True)
             self.guesser.loss = self.guesser.criterion(self.logits, y)        
             self.guesser.loss.backward()
             self.guesser.optimizer.step()
             # update learning rate
             self.guesser.update_learning_rate()

             
         return reward