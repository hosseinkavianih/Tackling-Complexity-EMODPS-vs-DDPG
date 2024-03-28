class Swmm_Model:
    """
    Adapted from pystorm library: https://github.com/pystorm/pystorm
    
    SUMMARY: This class is for running a swmm simulation allowing for
    real-time control; the main method is 'run simulation'
    
    INPUTS:
        config = .yaml filepath
        action_params = if the actions are from a parameterized function, these 
                        would be the parameters
    """
    
    def __init__(self, config, W = None, C = None, R = None, A = None, max_states = None, forecast = None):
        self.config = yaml.load(open(config, "r"), yaml.FullLoader)
        self.sim = Simulation(self.config["model_folder"] +
                              self.config["model_name"] +".inp") # initialize simulation
        self.sim.start()

        # methods
        self.methods = {
            "depthN": self._getNodeDepth,
            "depthL": self._getLinkDepth,
            "volumeN": self._getNodeVolume,
            "volumeL": self._getLinkVolume,
            "flow": self._getLinkFlow,
            "flooding": self._getNodeFlooding,
            "inflow": self._getNodeInflow,
            "setting": self._getValvePosition,
            "total_precip": self._getRainfall
        }
        
        self.W = W
        self.C = C
        self.R = R
        self.A = A
        self.max_states = max_states
        
        self.forecast = forecast
        
        # create datalog
        self.data_log = {"time":[],
                         "flow": {}, "inflow": {}, "flooding": {}, 'depthN':{}, 'setting':{}, 'total_precip': {}}
        
        if self.config["states_for_computing_objectives"] is not None:
            for entity, attribute in self.config["states_for_computing_objectives"]:
                self.data_log[attribute][entity] = []
    
        if self.config["states"] is not None:
            for entity, attribute in self.config["states"]:
                self.data_log[attribute][entity] = []
    
    def run_simulation(self):
        """
        purpose: 
            step simulation formward, applying actions; currently set up 
            to open and close valves
        output: 
            boolean indicating whether the simulation is finished
        """
        done = False
        self.step_dummy = 0
        while done == False:
            if self.W is not None and self.C is not None and self.R is not None and self.A is not None:
                actions = self._compute_actions()
                self._take_action(actions)
            time = self.sim._model.swmm_step()
            
            # log information
            self._log_tstep()
            if self._log_tstep() == None:
                self.step_dummy +=1
            self.data_log['time'].append(self.sim._model.getCurrentSimulationTime())
            
            done = False if time > 0 else True # time increases till the end of the sim., then resets to 0
        self._end_and_close()
    
    def _log_tstep(self):
        for attribute in self.data_log.keys():
            if attribute != "time" and len(self.data_log[attribute]) > 0:
                for entity in self.data_log[attribute].keys():# ID
                    self.data_log[attribute][entity].append(
                        self.methods[attribute](entity))
 
        
    # ------ Valve modifications -------------------------------------------
    def _getValvePosition(self, ID):
        return self.sim._model.getLinkResult(ID, tkai.LinkResults.setting.value)

    def _setValvePosition(self, ID, valve):
        self.sim._model.setLinkSetting(ID, valve)
        
    # ------ Node Parameters  ----------------------------------------------
    def _getNodeDepth(self, ID):
        return self.sim._model.getNodeResult(ID, tkai.NodeResults.newDepth.value)
    
    def _getRainfall(self, ID):
        return self.sim._model.getGagePrecip(ID, tkai.RainGageResults.total_precip.value)   

    def _getNodeFlooding(self, ID):
        return self.sim._model.getNodeResult(ID, tkai.NodeResults.overflow.value)

    def _getNodeLosses(self, ID):
        return self.sim._model.getNodeResult(ID, tkai.NodeResults.losses.value)
    
    def _getNodeVolume(self, ID):
        return self.sim._model.getNodeResult(ID, tkai.NodeResults.newVolume.value)

    def _getNodeInflow(self, ID):
        return self.sim._model.getNodeResult(ID, tkai.NodeResults.totalinflow.value)

    def _setInflow(self, ID, value):
        return self.sim._model.setNodeInflow(ID, value)

    # ------ Link modifications --------------------------------------------
    def _getLinkDepth(self, ID):
        return self.sim._model.getLinkResult(ID, tkai.LinkResults.newDepth.value)

    def _getLinkVolume(self, ID):
        return self.sim._model.getLinkResult(ID, tkai.LinkResults.newVolume.value)

    def _getLinkFlow(self, ID):
        return self.sim._model.getLinkResult(ID, tkai.LinkResults.newFlow.value)
        
    def _get_state(self):
        # create list of tuples with all state variables
        states = self.config["states_for_computing_objectives"].copy()
        states.extend(self.config["states"])
        
        state = []
        for s in states:
            entity = s[0] # name of object in swmm
            attribute = s[1] # attribute of interest (e.g. flow)
            state.append(self.methods[attribute](entity))

        state = np.asarray(state)
        
        return state
    

        
    def _get_lagged_states(self):
        states = self.config["lagged_state_variables"]
        lag_states = []        
        for s in states:
            entity = s[0] # name of object in swmm
            attribute = s[1] # attribute of interest (e.g. flow)
            for lag_hrs in self.config['lags_hrs']:
                ct = self.data_log['time'][-1]
                lagged_idx = 0
                dif = 0
                while dif < lag_hrs:
                    lagged_idx -= 1
                    lt = self.data_log['time'][lagged_idx]
                    dif = (ct - lt).total_seconds() / 60 / 60 # hours
                    
                lag_states.append(self.data_log[attribute][entity][lagged_idx])
                
        return lag_states
    
    def get_forecast(self):
    
        return self.forecast
    
    def step_current(self):
        return self.sim._model.getCurrentSimulationTime()
    
    
    
    def _compute_actions(self):
        # output actions; needs to be a list of equal length to 
        # self.config["action_space"]
        # this is where self.action_params would come into play
        # you'll also use the method self._state() here
                
        # Calculate normalized releases (u) corresponding to each sample of normalized inputs
        # u is a 2-D matrix with K columns (1 for each output)
        # and as many rows as there are samples of inputs
        #time1 = self.step_current()
        #date_str = time1.strftime("%Y-%m-%d")
        #hour_str = time1.strftime("%H:%M:%S")
        #datetime_str = date_str + ' ' + hour_str
        
        Forecast = self.get_forecast()
        
        #state_0 = Forecast['norfolk_airport_total_precip'][Forecast['time'] == datetime_str].values[0]
        
        normInputs = [Forecast[self.step_dummy]/self.max_states[0], self._getNodeDepth("P1")/self.max_states[1], self._getNodeDepth("P2")/self.max_states[2]]

        actions = [self.A[0], self.A[1]] # initialize actions at 0 for convex RBFs
        for k in range(K):
            for n in range(N):
                BF = 0
                for m in range(M):
                    if self.R[m,n] > 10**-6:
                        BF = BF + ((normInputs[m]-self.C[m,n])/self.R[m,n])**2
                    else:
                        BF = BF + ((normInputs[m]-self.C[m,n])/(10**-6))**2
                        
                actions[k] = actions[k] + self.W[k,n]*np.exp(-BF)

            # check if u between 0 and 1
            if actions[k] < 0.0:
                actions[k] = 0.0
            elif actions[k] > 1.0:
                actions[k] = 1.0
        
        #print(actions)
        return actions
    
    
    def export_df(self):
        for key in self.data_log.keys():
            if key == 'time':
                df = pd.DataFrame({key : self.data_log[key]})
                continue
            tmp = pd.DataFrame.from_dict(self.data_log[key])
            if len(tmp) == 0:
                continue
            new_col_names = []
            for col_name in tmp.columns:
                new_col_names.append(str(col_name) + '_' + key)
            tmp.columns = new_col_names
            
            df = df.merge(tmp, left_index=True, right_index=True)
            
        return df
        
    
    def _take_action(self, actions=None):
        if actions is not None:
            for entity, valve_position in zip(self.config["action_space"], actions):
                self._setValvePosition(entity, valve_position)
                
    def _end_and_close(self):
        """
        Terminates the simulation
        """
        self.sim._model.swmm_end()
        self.sim._model.swmm_close()
