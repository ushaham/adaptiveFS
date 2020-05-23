"""
Created on Wed Nov 13 21:20:04 2019

Code is based on the official tutorial in
https://gym.openai.com/evaluations/eval_onwKGm96QkO9tJwdX7L0Gw/
"""
import argparse
import os
import shutil
import torch
import torch.nn
from torch.optim import lr_scheduler
import numpy as np
import random
from collections import namedtuple
from collections import deque
from typing import List, Tuple
from itertools import count
from sklearn.metrics import confusion_matrix, roc_auc_score


from questionnaire_env import Questionnaire_env
from questionnaire_env import Guesser

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--save_dir",
                    type=str,
                    default='./ddqn_models',
                    help="Directory for saved models")
parser.add_argument("--gamma",
                    type=float,
                    default=0.95,
                    help="Discount rate for Q_target")
parser.add_argument("--n_update_target_dqn",
                    type=int,
                    default=10,
                    help="Mumber of episodes between updates of target dqn")
parser.add_argument("--val_trials_wo_im",
                    type=int,
                    default=50,
                    help="Number of validation trials without improvement")
parser.add_argument("--ep_per_trainee",
                    type=int,
                    default=1000,
                    help="Switch between training dqn and guesser every this # of episodes")
parser.add_argument("--batch_size",
                    type=int,
                    default=32,
                    help="Mini-batch size")
parser.add_argument("--hidden-dim",
                    type=int,
                    default=128,
                    help="Hidden dimension")
parser.add_argument("--capacity",
                    type=int,
                    default=1000,
                    help="Replay memory capacity")
parser.add_argument("--max-episode",
                    type=int,
                    default=2000,
                    help="e-Greedy target episode (eps will be the lowest at this episode)")
parser.add_argument("--min-eps",
                    type=float,
                    default=0.05,
                    help="Min epsilon")
parser.add_argument("--lr",
                    type=float,
                    default=1e-4,
                    help="Learning rate")
parser.add_argument("--min_lr",
                    type=float,
                    default=1e-6,
                    help="Minimal learning rate")
parser.add_argument("--decay_step_size",
                    type=int,
                    default=17500,
                    help="LR decay step size")
parser.add_argument("--lr_decay_factor",
                    type=float,
                    default=0.1,
                    help="LR decay factor")
parser.add_argument("--weight_decay",
                    type=float,
                    default=0.,
                    help="l_2 weight penalty")
parser.add_argument("--val_interval",
                    type=int,
                    default=1000,
                    help="Interval for calculating validation reward and saving model")
parser.add_argument("--episode_length",
                    type=int,
                    default=8,
                    help="Episode length")
parser.add_argument("--case",
                    type=int,
                    default=122,
                    help="Which data to use")
parser.add_argument("--env",
                    type=str,
                    default="Questionnaire",
                    help="environment name: Questionnaire")
#### environment params
parser.add_argument("--g_hidden-dim",
                    type=int,
                    default=256,
                    help="Guesser hidden dimension")
parser.add_argument("--g_weight_decay",
                    type=float,
                    default=0e-4,
                    help="l_2 weight penalty for guesser")

FLAGS = parser.parse_args(args=[])

#set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DQN(torch.nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int, 
                 hidden_dim: int) -> None:
        """DQN Network
        Args:
            input_dim (int): `state` dimension.
                `state` is 2-D tensor of shape (n, input_dim)
            output_dim (int): Number of actions.
                Q_value is 2-D tensor of shape (n, output_dim)
            hidden_dim (int): Hidden dimension in fc layer
        """
        super(DQN, self).__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )        

        if FLAGS.env == 'Questionnaire':
            self.final = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, output_dim),
                torch.nn.Sigmoid()
            )
        else:
            self.final = torch.nn.Linear(hidden_dim, output_dim)	

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns a Q_value
        Args:
            x (torch.Tensor): `State` 2-D tensor of shape (n, input_dim)
        Returns:
            torch.Tensor: Q_value, 2-D tensor of shape (n, output_dim)
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.final(x)

        return x

Transition = namedtuple("Transition",
                        field_names=["state", "action", "reward", "next_state", "done"])


class ReplayMemory(object):

    def __init__(self, capacity: int) -> None:
        """Replay memory class
        Args:
            capacity (int): Max size of this memory
        """
        self.capacity = capacity
        self.cursor = 0
        self.memory = []

    def push(self,
             state: np.ndarray,
             action: int,
             reward: int,
             next_state: np.ndarray,
             done: bool) -> None:
        """Creates `Transition` and insert
        Args:
            state (np.ndarray): 1-D tensor of shape (input_dim,)
            action (int): action index (0 <= action < output_dim)
            reward (int): reward value
            next_state (np.ndarray): 1-D tensor of shape (input_dim,)
            done (bool): whether this state was last step
        """
        if len(self) < self.capacity:
            self.memory.append(None)

        self.memory[self.cursor] = Transition(state,
                                              action, reward, next_state, done)
        self.cursor = (self.cursor + 1) % self.capacity

    def pop(self, batch_size: int) -> List[Transition]:
        """Returns a minibatch of `Transition` randomly
        Args:
            batch_size (int): Size of mini-bach
        Returns:
            List[Transition]: Minibatch of `Transition`
        """
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        """Returns the length """
        return len(self.memory)
    
def lambda_rule(i_episode) -> float:
    """ stepwise learning rate calculator """
    exponent = int(np.floor((i_episode + 1) / FLAGS.decay_step_size))
    return np.power(FLAGS.lr_decay_factor, exponent)


class Agent(object):

    def __init__(self, 
                 input_dim: int, 
                 output_dim: int, 
                 hidden_dim: int) -> None:
        """Agent class that choose action and train
        Args:
            input_dim (int): input dimension
            output_dim (int): output dimension
            hidden_dim (int): hidden dimension
        """
        self.dqn = DQN(input_dim, output_dim, hidden_dim)
        self.target_dqn = DQN(input_dim, output_dim, hidden_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.loss_fn = torch.nn.MSELoss()
        self.optim = torch.optim.Adam(self.dqn.parameters(), 
                                      lr=FLAGS.lr, 
                                      weight_decay=FLAGS.weight_decay)
        
        self.scheduler = lr_scheduler.LambdaLR(self.optim, 
                                               lr_lambda=lambda_rule)
        
        self.update_target_dqn()
        

    def update_target_dqn(self):
        
        # hard copy model parameters to target model parameters
        for param, target_param in zip(self.dqn.parameters(), self.target_dqn.parameters()):
            target_param.data.copy_(param.data)

    def _to_variable(self, x: np.ndarray) -> torch.Tensor:
        """torch.Variable syntax helper
        Args:
            x (np.ndarray): 2-D tensor of shape (n, input_dim)
        Returns:
            torch.Tensor: torch variable
        """
        return torch.autograd.Variable(torch.Tensor(x))

    def get_action(self, states: np.ndarray, 
                   eps: float,
                   mask: np.ndarray) -> int:
        """Returns an action
        Args:
            states (np.ndarray): 2-D tensor of shape (n, input_dim)
            eps (float): 𝜺-greedy for exploration
            mask (np.ndarray) zeroes out q values for questions that were already asked, so they will not be chosen again
        Returns:
            int: action index
        """
        if np.random.rand() < eps:
            #return np.random.choice(self.output_dim)
            return np.random.choice(self.output_dim, p=env.action_probs)
        else:
            self.dqn.train(mode=False)
            scores = self.get_Q(states)
            _, argmax = torch.max(scores.data * mask, 1)
            return int(argmax.item())

    def get_Q(self, states: np.ndarray) -> torch.FloatTensor:
        """Returns `Q-value`
        Args:
            states (np.ndarray): 2-D Tensor of shape (n, input_dim)
        Returns:
            torch.FloatTensor: 2-D Tensor of shape (n, output_dim)
        """
        states = self._to_variable(states.reshape(-1, self.input_dim))
        states = states.to(device) 
        self.dqn.train(mode=False)
        return self.dqn(states)
    
    def get_target_Q(self, states: np.ndarray) -> torch.FloatTensor:
        """Returns `Q-value`
        Args:
            states (np.ndarray): 2-D Tensor of shape (n, input_dim)
        Returns:
            torch.FloatTensor: 2-D Tensor of shape (n, output_dim)
        """
        states = self._to_variable(states.reshape(-1, self.input_dim))
        self.target_dqn.train(mode=False)
        return self.target_dqn(states)

    def train(self, Q_pred: torch.FloatTensor, Q_true: torch.FloatTensor) -> float:
        """Computes `loss` and backpropagation
        Args:
            Q_pred (torch.FloatTensor): Predicted value by the network,
                2-D Tensor of shape(n, output_dim)
            Q_true (torch.FloatTensor): Target value obtained from the game,
                2-D Tensor of shape(n, output_dim)
        Returns:
            float: loss value
        """
        self.dqn.train(mode=True)
        self.optim.zero_grad()
        loss = self.loss_fn(Q_pred, Q_true)
        loss.backward()
        self.optim.step()

        return loss
    
    def update_learning_rate(self):
        """ Learning rate updater """
        
        self.scheduler.step()
        lr = self.optim.param_groups[0]['lr']
        if lr < FLAGS.min_lr:
            self.optim.param_groups[0]['lr'] = FLAGS.min_lr
            lr = self.optim.param_groups[0]['lr']
        print('DQN learning rate = %.7f' % lr)


def train_helper(agent: Agent, 
                 minibatch: List[Transition], 
                 gamma: float) -> float:
    """Prepare minibatch and train them
    Args:
        agent (Agent): Agent has `train(Q_pred, Q_true)` method
        minibatch (List[Transition]): Minibatch of `Transition`
        gamma (float): Discount rate of Q_target
    Returns:
        float: Loss value
    """
    states = np.vstack([x.state for x in minibatch])
    actions = np.array([x.action for x in minibatch])
    rewards = np.array([x.reward for x in minibatch])
    next_states = np.vstack([x.next_state for x in minibatch])
    done = np.array([x.done for x in minibatch])

    Q_predict = agent.get_Q(states)
    Q_target = Q_predict.clone().cpu().data.numpy()
    max_actions = np.argmax(agent.get_Q(next_states).data.cpu().numpy(), axis=1)
    Q_target[np.arange(len(Q_target)), actions] = rewards + gamma * agent.get_target_Q(next_states)[np.arange(len(Q_target)), max_actions].data.numpy() * ~done
    Q_target = agent._to_variable(Q_target).to(device=device)

    return agent.train(Q_predict, Q_target)


def play_episode(env,
                 agent: Agent,
                 replay_memory: ReplayMemory,
                 eps: float,
                 batch_size: int,
                 train_guesser=True,
                 train_dqn=True) -> int:
    """Play an epsiode and train
    Args:
        env (gym.Env): gym environment (CartPole-v0)
        agent (Agent): agent will train and get action
        replay_memory (ReplayMemory): trajectory is saved here
        eps (float): 𝜺-greedy for exploration
        batch_size (int): batch size
    Returns:
        int: reward earned in this episode
    """
    s = env.reset(train_guesser=train_guesser)
    done = False
    total_reward = 0
    mask = env.reset_mask()

    t = 0
    while not done:

        a = agent.get_action(s, eps, mask)
        s2, r, done, info = env.step(a)
        mask[a] = 0

        total_reward += r
                
        replay_memory.push(s, a, r, s2, done)

        if len(replay_memory) > batch_size:
            
            if train_dqn:
                minibatch = replay_memory.pop(batch_size)
                train_helper(agent, minibatch, FLAGS.gamma)

        s = s2
        t += 1
        
        if t == FLAGS.episode_length:
            break
            
    if train_dqn:
        agent.update_learning_rate()

    return total_reward, t


def get_env_dim(env) -> Tuple[int, int]:
    
    """Returns input_dim & output_dim
    Args:
        env (gym.Env): gym Environment
    Returns:
        int: input_dim
        int: output_dim
    """
    input_dim = 2 * env.n_questions
    output_dim = env.n_questions  + 1  
        
    return input_dim, output_dim


def epsilon_annealing(episode: int, max_episode: int, min_eps: float) -> float:
    """Returns 𝜺-greedy
    1.0---|\
          | \
          |  \
    min_e +---+------->
              |
              max_episode
    Args:
        epsiode (int): Current episode (0<= episode)
        max_episode (int): After max episode, 𝜺 will be `min_eps`
        min_eps (float): 𝜺 will never go below this value
    Returns:
        float: 𝜺 value
    """

    slope = (min_eps - 1.0) / max_episode
    return max(slope * episode + 1.0, min_eps)


# define envurinment and agent (needed for main and test)
env = Questionnaire_env(flags=FLAGS,
                        device=device)
clear_threshold = 1.
    
# define agent   
input_dim, output_dim = get_env_dim(env) 
agent = Agent(input_dim, 
              output_dim, 
              FLAGS.hidden_dim)    

agent.dqn.to(device=device)
env.guesser.to(device=device)
    
def save_networks(i_episode: int, 
                  val_auc=None) -> None:
    """ A method to save parameters of guesser and dqn """
    if not os.path.exists(FLAGS.save_dir):
        os.makedirs(FLAGS.save_dir)
    
    if i_episode == 'best':
        guesser_filename = 'best_guesser.pth'
        dqn_filename = 'best_dqn.pth'
    else:
        guesser_filename = '{}_{}_{:1.3f}.pth'.format(i_episode, 'guesser', val_auc)
        dqn_filename = '{}_{}_{:1.3f}.pth'.format(i_episode, 'dqn', val_auc)
        
    guesser_save_path = os.path.join(FLAGS.save_dir, guesser_filename)
    dqn_save_path = os.path.join(FLAGS.save_dir, dqn_filename)
    
    # save guesser
    if os.path.exists(guesser_save_path):
        os.remove(guesser_save_path)
    torch.save(env.guesser.cpu().state_dict(), guesser_save_path + '~')
    env.guesser.to(device=device)
    os.rename(guesser_save_path + '~', guesser_save_path)
    
    # save dqn
    if os.path.exists(dqn_save_path):
        os.remove(dqn_save_path)
    torch.save(agent.dqn.cpu().state_dict(), dqn_save_path + '~')
    agent.dqn.to(device=device)
    os.rename(dqn_save_path + '~', dqn_save_path)
    
def load_networks(i_episode: int, 
                  val_auc=None) -> None:
    """ A method to load parameters of guesser and dqn """
    if i_episode == 'best':
        guesser_filename = 'best_guesser.pth'
        dqn_filename = 'best_dqn.pth'
    else:
        guesser_filename = '{}_{}_{:1.3f}.pth'.format(i_episode, 'guesser', val_auc)
        dqn_filename = '{}_{}_{:1.3f}.pth'.format(i_episode, 'dqn', val_auc)
        
    guesser_load_path = os.path.join(FLAGS.save_dir, guesser_filename)
    dqn_load_path = os.path.join(FLAGS.save_dir, dqn_filename)
    
    # load guesser
    guesser = Guesser(state_dim=2 * env.n_questions,
                      hidden_dim=FLAGS.g_hidden_dim,
                      lr=FLAGS.lr,
                      min_lr=FLAGS.min_lr,
                      weight_decay=FLAGS.g_weight_decay,
                      decay_step_size=FLAGS.decay_step_size,
                      lr_decay_factor=FLAGS.lr_decay_factor)
 
    
    guesser_state_dict = torch.load(guesser_load_path)
    guesser.load_state_dict(guesser_state_dict)
    guesser.to(device=device)
    
    # load sqn
    dqn = DQN(input_dim, output_dim, FLAGS.hidden_dim)
    dqn_state_dict = torch.load(dqn_load_path)
    dqn.load_state_dict(dqn_state_dict)
    dqn.to(device=device)
    
    return guesser, dqn
    
def main():
    """ Main """
    
    # delete model files from previous runs
    if os.path.exists(FLAGS.save_dir):
        shutil.rmtree(FLAGS.save_dir)
        
    # store best result
    best_val_auc = 0
    
    # counter of validation trials with no improvement, to determine when to stop training
    val_trials_without_improvement = 0
    
    # set up trainees for first cycle
    train_guesser = False
    train_dqn = True
    
    rewards = deque(maxlen=100)
    steps = deque(maxlen=100)
    
   
    replay_memory = ReplayMemory(FLAGS.capacity)

    for i in count(1):
        
        # determint whether gesser or dqn is trained
        if i % (2 * FLAGS.ep_per_trainee) == FLAGS.ep_per_trainee:
            train_dqn = False
            train_guesser = True
        if i % (2 * FLAGS.ep_per_trainee) == 0:
            train_dqn = True
            train_guesser = False
            
        # set exploration epsilon
        eps = epsilon_annealing(i, FLAGS.max_episode, FLAGS.min_eps)
        
        # play an episode
        r, t = play_episode(env, 
                            agent, 
                            replay_memory, 
                            eps, 
                            FLAGS.batch_size,
                            train_dqn=train_dqn,
                            train_guesser=train_guesser)
        
        # store rewards and episode length
        rewards.append(r)
        steps.append(t)
        
        # print results to console
        print("[Episode: {:5}], Steps: {}, Avg steps: {:1.3f}, Reward: {:1.3f}, Avg reward: {:1.3f}, 𝜺-greedy: {:5.2f}".format(i, t, np.mean(steps), r, np.mean(rewards), eps))
        
        # check if environment is solved
        if len(rewards) == rewards.maxlen:
            if np.mean(rewards) >= clear_threshold:
                print("Environment solved in {} episodes with {:1.3f}".format(i, np.mean(rewards)))
                break
        
        if i % FLAGS.val_interval == 0:
            # compute performance on validation set
            new_best_val_auc = val(i_episode=i, 
                                   best_val_auc=best_val_auc)
            
            # update best result on validation set and counter
            if new_best_val_auc > best_val_auc:
                best_val_auc = new_best_val_auc
                val_trials_without_improvement = 0
            else:
                val_trials_without_improvement += 1
                
        # check whether to stop training
        if val_trials_without_improvement == FLAGS.val_trials_wo_im:
            print('Did not achieve val AUC improvement for {} trials, training is done.'.format(FLAGS.val_trials_wo_im))
            break
        
        if i % FLAGS.n_update_target_dqn == 0:
                agent.update_target_dqn()
                
            
def val(i_episode: int, 
        best_val_auc: float) -> float:
    """ Compute performance on validation set and save current models """
    
    print('Running validation')
    
    # placeholder to store outcome probabilities
    y_hat_val_prob = np.zeros(len(env.y_val))
    
    # perform an episode on each validation patient
    for i in range(len(env.X_val)):
        
        # reset reward, state and mask
        ep_reward = 0
        state = env.reset(mode='val', 
                          patient=i,
                          train_guesser=False)
        mask = env.reset_mask()
        
        # run episode
        for t in range(FLAGS.episode_length):

            # select action from policy
            action = agent.get_action(state, eps=0, mask=mask)
            mask[action] = 0
            
            # take the action
            state, reward, done, guess = env.step(action, mode='val')
            
            # if a guess was made, record the outcome probability
            if guess != -1:
                y_hat_val_prob[i] = env.outcome_prob
            
            # update reward
            ep_reward += reward
             
            if done:
                break
    
    # update AUROC score
    roc_auc_score_ = roc_auc_score(env.y_val,  y_hat_val_prob)
    
    
    # Save best performing model, if needed
    if roc_auc_score_ > best_val_auc:
        print('New best AUC acheievd, saving best model')
        save_networks(i_episode='best')
        save_networks(i_episode, roc_auc_score_)
        
        return roc_auc_score_
    
    else:
        return best_val_auc
        
                
            
def test():
    """ Computes performance nad test data """
    
    # load best performing models
    print('Loading best networks')
    env.guesser, agent.dqn = load_networks(i_episode='best')

    # predict outcome on test data
    y_hat_test = np.zeros(len(env.y_test))
    y_hat_test_prob = np.zeros(len(env.y_test))
    
    print('Computing predictions of test data')
    n_test = len(env.X_test)
    for i in range(n_test):
        
        if i % 10000 == 0:
            print('{} / {}'.format(i, n_test))
        
        # reset state and mask
        state = env.reset(mode='test', 
                          patient=i,
                          train_guesser=False)
        mask = env.reset_mask()
        
        # run episode
        for t in range(FLAGS.episode_length):

            # select action from policy
            action = agent.get_action(state, eps=0, mask=mask)
            mask[action] = 0
            
            # take the action
            state, reward, done, guess = env.step(action, mode='test') 
            
            if guess != -1:
                y_hat_test_prob[i] = env.outcome_prob
                                           
            if done:
                break
        y_hat_test[i] = guess
           
    # Compute confusion matrix and AUROC score
    C = confusion_matrix(env.y_test, y_hat_test)
    print('Confusion matrix: ')
    # If C is 3x3, the first column corresponds to "no guess was made"
    print(C)

    class_0_recall = C[-2,-2] / sum(C[-2,:])
    class_1_recall = C[-1,-1] / sum(C[-1,:])
    class_0_precision = C[-2,-2] / sum(C[:,-2])
    class_1_precision = C[-1,-1] / sum(C[:,-1])
    roc_auc_score_ = roc_auc_score(env.y_test,  y_hat_test_prob)
    
    print('class 0 recall: ', np.round(class_0_recall, 3))
    print('class 1 recall: ', np.round(class_1_recall, 3))
    print('class 0 precision: ', np.round(class_0_precision, 3))
    print('class 1 precision: ', np.round(class_1_precision, 3))
    print('AUROC: ', np.round(roc_auc_score_, 3))
    
def show_sample_paths(n_patients=1):
    """A method to run episodes on randomly chosen positive and negative test patients, and print trajectories to console  """
    
    # load best performing networks
    print('Loading best networks')
    env.guesser, agent.dqn = load_networks(i_episode='best')
    
    for i in range(n_patients):
        print('Starting new episode with a new test patient')
        # Randomly choose a positive test patient, and recover original feature values
        pos_patient = random.choice(env.test_class_1_inds)
        feature_vec = (env.X_test[pos_patient].reshape(1, -1) + 1) / 2.
        feature_vec = env.scaler.inverse_transform(feature_vec).squeeze()
        
        # Reset state and mask
        state = env.reset(mode='test', 
                          patient=pos_patient,
                          train_guesser=False)
        mask = env.reset_mask()
        
        print('Basic info: sex: {}, age: {}, race:{}'.format(int(feature_vec[env.sex_var]), int(feature_vec[env.age_var]), int(feature_vec[env.race_vars])))
        
        # run episode
        for t in range(FLAGS.episode_length):

            # select action from policy
            action = agent.get_action(state, eps=0, mask=mask)
            mask[action] = 0
            
            if action != env.n_questions:
                print('Step: {}, Question: '.format(t + 1), env.question_names[action], ', Answer: {:1.2f}'.format(feature_vec[action]))
            
            
            # take the action
            state, reward, done, guess = env.step(action, mode='test') 
            
            if guess != -1:
                print('Step: {}, Ready to make a guess: Prob(y=1)={:1.3f}, Guess: y={}, Ground truth: y=1'.format(t + 1, env.outcome_prob, guess))
                                           
            if done:
                print('Episode terminated\n')
                break
            
        print('Starting new episode with a new test patient')
        # Randomly choose a negative patient and revover original feature values
        neg_patient = random.choice(env.test_class_0_inds)
        feature_vec = (env.X_test[neg_patient].reshape(1, -1) + 1) / 2
        feature_vec = env.scaler.inverse_transform(feature_vec).squeeze()

        # Reset state and mask
        state = env.reset(mode='test', 
                          patient=neg_patient,
                          train_guesser=False)
        mask = env.reset_mask()
        
        print('Basic info: sex: {}, age: {}, race:{}'.format(int(feature_vec[env.sex_var]), int(feature_vec[env.age_var]), int(feature_vec[env.race_vars])))
        
        # run episode
        for t in range(FLAGS.episode_length):

            # select action from policy
            action = agent.get_action(state, eps=0, mask=mask)
            mask[action] = 0
            
            if action != env.n_questions:
                print('Step: {}, Question: '.format(t + 1), env.question_names[action], ', Answer: {:1.2f}'.format(feature_vec[action]))
            
            
            # take the action
            state, reward, done, guess = env.step(action, mode='test') 
            
            if guess != -1:
                print('Step: {}, Ready to make a guess: Prob(y=1)={:1.3f}, Guess: y={}, Ground truth: y=0'.format(t + 1, env.outcome_prob, guess))
                                           
            if done:
                print('Episode terminated\n')
                break

def predict_interactively():
    """ Making predictions in interactive fashion """
    
    print('Loading best networks')
    env.guesser, agent.dqn = load_networks(i_episode='best')
    
    state = env.reset(mode='interactive', 
                          train_guesser=False)
    mask = env.reset_mask()
    
    # run episode
    for t in range(FLAGS.episode_length):

        # select action from policy
        action = agent.get_action(state, eps=0, mask=mask)
        mask[action] = 0
        print('Step {}: '.format(t+1))
        # take the action
        state, reward, done, guess = env.step(action, mode='interactive') 
        
        if guess != -1:
            print('Ready to make a guess: Prob(y=1)={:1.3f}, Guess: y={}'.format(env.outcome_prob, guess))
                                       
        if done:
            print('Episode terminated\n')
            break
    
if __name__ == '__main__':
    main()
    test()
    show_sample_paths(2)
    # predict_interactively()
