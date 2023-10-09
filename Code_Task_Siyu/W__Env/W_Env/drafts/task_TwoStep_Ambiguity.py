from W_Gym.W_Gym import W_Gym

class task_TwoStep_Ambiguity_1frame(W_Gym):
    task_param = {}
    high_state = None
    is_fixed = False

    def get_versionname(self):
        tstr = 'T' if self.task_hyper_param['is_random_common0'] else 'F'
        return f"pR{self.task_hyper_param['ps_high_state']*100:.0f}_pSR{self.task_hyper_param['p_switch_reward']*1000:.0f}_pT{self.task_hyper_param['ps_common_trans']*100:.0f}_pST{self.task_hyper_param['p_switch_transition']*1000:.0f}_PST0{tstr}_pA{self.task_hyper_param['ps_ambiguity']*100:.0f}"
        

    def _block_info(self):
        self.info_block['params'] = self.task_param

    def _trial_info(self):
        self.info_trial['params'] = self.param_trial 



    def format4save(self):
        d = super().format4save()
        d.obs = np.matmul(np.stack(d.obs), np.array([[0],[1],[2]])).squeeze()
        d.obs_next = np.matmul(np.stack(d.obs_next), np.array([[0],[1],[2]])).squeeze()
        # d.trialID = np.ceil(d.tot_t/2)
        d.stage = 1 - d.stage
        d.transition += 1
        return d