os.chdir("C:/Users/test/Dropbox/tml/IHS/simu")
sys.path.append("C:/Users/test/Dropbox/tml/IHS/simu") 

M = 1
method_list = ['overall']
signal_factor_list = [0.5]
train_episodes = 0
test_size_factor = 1
for signal_factor in signal_factor_list:
  for method in method_list:
      for seed in range(M):
          arg_pass = str(seed) + ' '+ str(signal_factor)+ ' '+ method + ' ' + str(train_episodes) +' ' + str(test_size_factor)
          runfile('simu/dqn_offline.py',args=arg_pass)
