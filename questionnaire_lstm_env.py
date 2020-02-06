# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 21:49:57 2019

@author: urixs

Environment for questionnaire

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
    
    """
    implements LSTM state update mechanism
    """
    
    def __init__(self, n_questions):
        super(Guesser, self).__init__()
        
        self.embedding_dim = 128
        self.state_dim = 256
        
        # question embedding, we add one a "dummy question" for cases when guess is made at first step
        self.q_emb = nn.Embedding(num_embeddings=n_questions + 1, 
                                  embedding_dim=self.embedding_dim)

        input_dim = 2 * self.embedding_dim
        self.lstm = nn.LSTMCell(input_size=input_dim, hidden_size=self.state_dim)
        self.affine = nn.Linear(self.state_dim, 2)
        self.reset_states()
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.optimizer = torch.optim.Adam(self.parameters(), 	
                                          weight_decay=FLAGS.weight_decay,	
                                          lr=FLAGS.lr)	
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, 	
                                               lr_lambda=lambda_rule)
     
    def forward(self, question, answer):
        question_embedding = self.q_emb(torch.LongTensor([question]))
        answer_vec = torch.unsqueeze(torch.ones(self.embedding_dim) * answer, 0)
        x = torch.cat((question_embedding, 
                       answer_vec), dim=-1)
        self.lstm_h, self.lstm_c = self.lstm(x, (self.lstm_h, self.lstm_c))  
        logits = self.affine(self.lstm_h)
        probs = F.softmax(logits, dim=1)
        return self.lstm_h, logits, probs
    
    def reset_states(self):
        self.lstm_h = torch.zeros(1, self.state_dim)
        self.lstm_c = torch.zeros(1, self.state_dim)

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


class Questionnaire_env(gym.Env):
     """ Questionnaire Environment class
        Args:
            case (int): which data to use
            oversample (Boolean): whether to oversample the small class
            load_pretrained_guesser (Boolean): whether to load a pretrained guesser
        """
        
     def __init__(self, 
                  case=1, 
                  oversample=True,
                  episode_length=5):
         
         # Load data
         X, y, question_names, class_names, self.scaler  = utils.load_data(case)
         self.n_questions = X.shape[1]
         self.question_names = question_names
         
         # Identify columns of preliminay info variables
         self.sex_var = np.argmax(self.question_names == 'sex')
         self.age_var = np.argmax(self.question_names == 'age_p')
         self.race_vars = np.argmax(self.question_names == 'hiscodi32')
         
         self.episode_length = episode_length
         
         # Randomly split data to train, validation and test sets
         self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, 
                                                                                 y, 
                                                                                 test_size=0.33, 
                                                                                 random_state=42)
         self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, 
                                                                                self.y_train, 
                                                                                test_size=0.05, 
                                                                                random_state=24)
         if len(self.X_val > 5000):
             self.X_val = self.X_val[:5000]
             self. y_val = self.y_val[:5000]

        
        # Store indices of positive and negative patients
         self.class_0_inds = [index for index,value in enumerate(self.y_train) if value == 0]
         self.class_1_inds = [index for index,value in enumerate(self.y_train) if value == 1]
         self.test_class_0_inds = [index for index,value in enumerate(self.y_test) if value == 0]
         self.test_class_1_inds = [index for index,value in enumerate(self.y_test) if value == 1]
         
         self.oversample = oversample
         
         self.current_class = 0
         
         self.action_translation = {self.n_questions: "guess does not die", 
                                    self.n_questions + 1: "guess dies"}
         
         self.guesser = Guesser(2 * self.n_questions)
         
         print('Initialized LSTM-questionnaire environment')                  
     
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
         
         # Reset state
         self.guesser.reset_states()
         self.state = self.guesser.lstm_h.detach().numpy()
        
         if  mode == 'training':
             if not self.oversample:
                 self.patient = np.random.randint(self.X_train.shape[0])
             else:
                 if np.random.rand() < .5:
                     ind = np.random.randint(len(self.class_0_inds))
                     self.patient = self.class_0_inds[ind]
                     self.current_class = 0
                 else:
                     ind = np.random.randint(len(self.class_1_inds))
                     self.patient = self.class_1_inds[ind]
                     self.current_class  = 1                
         else: 
             self.patient = patient
             
             
         # obtain patient feature vec
         if mode == 'training':
             patient_vec = self.X_train[self.patient]
         if mode == 'val':
             patient_vec = self.X_val[self.patient]
         if mode == 'test':
             patient_vec = self.X_test[self.patient]
             
         # update state with sex, age and race info
         if self.sex_var != None:
             action = self.sex_var
             answer = patient_vec[self.sex_var]
             _, _, _ = self.guesser(action, answer) 
             
         if self.age_var != None:
             action = self.age_var
             answer = patient_vec[self.age_var]
             _, _, _ = self.guesser(action, answer) 
             
         if self.race_vars != None:
             action = self.race_vars
             answer = patient_vec[self.race_vars]
             _, _, _ = self.guesser(action, answer) 
             
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
        
         # update mask with sex, age and race entries set to zero
         if self.sex_var != None:
             mask[self.sex_var] = 0
         if self.age_var != None:
             mask[self.age_var] = 0
         if self.race_vars != None:
             mask[self.race_vars] = 0
         
         return mask
             
         
     def step(self, 
              action, 
              mode='training'):
         """ State update mechanism """
         
         # update state
         next_state = self.update_state(action, mode)
         self.state = np.array(next_state)     
         self.s = np.array(self.state)
          
         '''        
         # compute reward
         self.reward = self.compute_reward(mode)
         '''
         self.time += 1
         if self.time == self.episode_length:
             self.terminate_episode()
             
         # compute reward
         self.reward = self.compute_reward(mode)
        
         return self.s, self.reward, self.done, self.guess
     
     # Update 'done' flag when episode terminates   
     def terminate_episode(self):
         self.done = True
            
     def update_state(self, action, mode):
         
         if action < self.n_questions: # we are not making a guess
             if  mode == 'training':
                 answer = self.X_train[self.patient, action]        
             elif mode == 'val':
                 answer = self.X_val[self.patient, action]
             elif mode == 'test': 
                 answer = self.X_test[self.patient, action]
             next_state,  self.logits, self.probs = self.guesser(action, answer)   
             next_state = next_state.detach().numpy()
             self.guess = -1
             self.done = False
             self.outcome_prob = self.probs.detach().numpy().squeeze()[1]
             if mode == 'training':
                 self.correct_prob = self.probs.detach().numpy().squeeze()[self.y_train[self.patient]]
             if mode == 'val':
                 self.correct_prob = self.probs.detach().numpy().squeeze()[self.y_val[self.patient]]
             
         else: # making a guess
              # "dummy" forward in case a guess was made at first step, to fill buffers
              if self.time == 0:
                  _,  self.logits, self.probs = self.guesser(question=self.n_questions, answer=0) 
              self.guess = np.argmax(self.probs.detach().numpy().squeeze())
              '''
              self.outcome_prob = self.probs.detach().numpy().squeeze()[1]
              if mode == 'training':
                  self.correct_prob = self.probs.detach().numpy().squeeze()[self.y_train[self.patient]]
              if mode == 'val':
                  self.correct_prob = self.probs.detach().numpy().squeeze()[self.y_val[self.patient]]
              '''
              self.terminate_episode()
              next_state = self.state
             
         return next_state
     
     def compute_reward(self, mode):
         """ Compute the reward """
         
         if mode == 'test':
             return None
         if mode == 'training':
             y_true = self.y_train[self.patient]
         
         if self.guess == -1: # no guess was made
             reward = .01 * np.random.rand()
         else:
             reward = self.correct_prob
         if self.train_guesser and self.done:
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