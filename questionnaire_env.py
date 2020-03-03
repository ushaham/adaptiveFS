# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 21:49:57 2019

@author: urixs

Environment for questionnaire

"""
import numpy as np
import os
from sklearn.model_selection import train_test_split
import gym
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.nn.functional as F

import utils

class Guesser(nn.Module):
    """
    implements a net that guesses the outcome given the state
    """
    
    def __init__(self,
                 state_dim, 
                 hidden_dim,
                 num_classes=2,
                 lr=1e-4,
                 min_lr=1e-6,
                 weight_decay=0.,
                 decay_step_size=12500,
                 lr_decay_factor=0.1):
        
        self.min_lr = min_lr
        
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
                                          weight_decay=weight_decay,
                                          lr=lr)
         
        self.lambda_rule = lambda x: np.power(lr_decay_factor, int(np.floor((x + 1) / decay_step_size)))
        
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, 
                                               lr_lambda=self.lambda_rule)


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
        if lr < self.min_lr:
            self.optimizer.param_groups[0]['lr'] = self.min_lr
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
                  flags,
                  device, 
                  oversample=True,
                  load_pretrained_guesser=True):
         
         case = flags.case
         episode_length = flags.episode_length
         self.device = device
         
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
         if len(self.X_val > 20000):
             self.X_val = self.X_val[:20000]
             self. y_val = self.y_val[:20000]
         
         # Compute correlations with target   
         correls = np.zeros(self.n_questions + 1)
         for i in range(self.n_questions):
             correls[i] = np.abs(np.corrcoef(self.y_train, self.X_train[:,i])[0,1])
         correls[self.n_questions] = .1
         self.action_probs = correls / np.sum(correls)

        
         # Store indices of positive and negative patients
         self.class_0_inds = [index for index,value in enumerate(self.y_train) if value == 0]
         self.class_1_inds = [index for index,value in enumerate(self.y_train) if value == 1]
         self.test_class_0_inds = [index for index,value in enumerate(self.y_test) if value == 0]
         self.test_class_1_inds = [index for index,value in enumerate(self.y_test) if value == 1]
         
         self.oversample = oversample
         
         self.current_class = 0
         
         self.action_translation = {self.n_questions: "guess does not die", 
                                    self.n_questions + 1: "guess dies"}
                  
         self.guesser = Guesser(state_dim=2 * self.n_questions,
                                hidden_dim=flags.g_hidden_dim,
                                lr=flags.lr,
                                min_lr=flags.min_lr,
                                weight_decay=flags.g_weight_decay,
                                decay_step_size=flags.decay_step_size,
                                lr_decay_factor=flags.lr_decay_factor)
         
         # Load pre-trained guesser network, if needed
         if load_pretrained_guesser:
             save_dir = './pretrained_guesser_models'
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
         Args: mode: training / val / test / interactive
               patient (int): index of patient
               train_guesser (Boolean): flag indicating whether to train guesser network in this episode
         
         Selects a patient (random for training, or pre-defined for val and test) ,
         Resets the state to contain the basic information,
         Resets 'done' flag to false,
         Resets 'train_guesser' flag
         """
         
         self.state =  np.zeros(2 * self.n_questions)
        
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
         if mode == 'interactive':
             patient_vec = None
             
         # update state with sex, age and race info
         if mode == 'interactive':
             sex = input('Please enter your sex (1=Male, 2=Female): \n')
             sex = float(sex)
             sex_scaled = utils.scale_individual_value(sex, self.sex_var, self.scaler)
             
             age = input('Please enter your age:\n')
             age = float(age)
             age_scaled = utils.scale_individual_value(age, self.age_var, self.scaler)
             
             race = input('Please enter your race (0=Non-Hispanic, 1=Hispanic):\n')
             race = float(race)
             race_scaled = utils.scale_individual_value(race, self.race_vars, self.scaler)
             
             self.state[self.sex_var] = sex_scaled
             self.state[self.sex_var + self.n_questions] = 1.
             self.state[self.age_var] = age_scaled
             self.state[self.age_var + self.n_questions] = 1.
             self.state[self.race_vars] = race_scaled
             self.state[self.race_vars + self.n_questions] = 1.                    
             
         else:
             if self.sex_var != None:
                 self.state[self.sex_var] = patient_vec[self.sex_var]
                 self.state[self.sex_var + self.n_questions] = 1.
             if self.age_var != None:
                 self.state[self.age_var] = patient_vec[self.age_var]
                 self.state[self.age_var + self.n_questions] = 1.
             if self.race_vars != None:
                 self.state[self.race_vars] = patient_vec[self.race_vars]
                 self.state[self.race_vars + self.n_questions] = 1.
         
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
             
         mask = mask.to(device=self.device)
         
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
             if mode == 'interactive':
                 str_ = 'Please enter {} :\n'.format(self.question_names[action])
                 ans = input(str_)
                 ans = float(ans)
                 next_state[action] = utils.scale_individual_value(ans, action, self.scaler)
             else:
                 if  mode == 'training':
                     next_state[action] = self.X_train[self.patient, action]        
                 elif mode == 'val':
                     next_state[action] = self.X_val[self.patient, action]
                 elif mode == 'test': 
                     next_state[action] = self.X_test[self.patient, action]
             next_state[action + self.n_questions] += 1.             
             self.guess = -1
             self.done = False
             
         else: # Making a guess
              # run guesser, and store guess and outcome probability
              guesser_input = self.guesser._to_variable(self.state.reshape(-1, 2 * self.n_questions))
              guesser_input = guesser_input.to(device=self.device)
              self.guesser.train(mode=False)
              self.logits, self.probs = self.guesser(guesser_input)
              self.guess = torch.argmax(self.probs.squeeze()).item()
              self.outcome_prob = self.probs.squeeze()[1].item()
              if mode == 'training':
                  # store probability of true outcome for reward calculation
                  self.correct_prob = self.probs.squeeze()[self.y_train[self.patient]].item()
              self.terminate_episode()
             
         return next_state
            
     def compute_reward(self, mode):
         """ Compute the reward """
         
         if mode == 'test':
             return None
         if mode == 'interactive':
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
             y = y.to(device = self.device)
             self.guesser.train(mode=True)
             self.guesser.loss = self.guesser.criterion(self.logits, y)        
             self.guesser.loss.backward()
             self.guesser.optimizer.step()
             # update learning rate
             self.guesser.update_learning_rate()

             
         return reward