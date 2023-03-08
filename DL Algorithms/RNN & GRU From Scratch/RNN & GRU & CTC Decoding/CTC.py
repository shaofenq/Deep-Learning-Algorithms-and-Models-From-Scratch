import numpy as np


class CTC(object):
    def __init__(self, BLANK=0):
        """
        
        Initialize instance variables

        Argument(s)
        -----------
        
        BLANK (int, optional): blank label index. Default 0.

        """

        # No need to modify
        self.BLANK = BLANK


    def extend_target_with_blank(self, target):
        extended_symbols = [self.BLANK]
        for symbol in target:
            extended_symbols.append(symbol)
            extended_symbols.append(self.BLANK)
            
        N = len(extended_symbols)
        skip_connect = np.zeros(N)
        skip_connect = np.zeros(N)
        for i, sym in enumerate(extended_symbols):
            if i>= 1 and (i< N-1) and (sym == self.BLANK) and (extended_symbols[i-1]!= extended_symbols[i+1]):
                skip_connect[i+1] = 1
        
  
        

        extended_symbols = np.array(extended_symbols).reshape((N,))
        skip_connect = np.array(skip_connect).reshape((N,))

        return extended_symbols, skip_connect
        #raise NotImplementedError
    def get_forward_probs(self, logits, extended_symbols, skip_connect):
        """Compute forward probabilities.

        Input
        -----
        logits: (np./Users/qinshaofeng/Desktop/HW3/handout/mytorch/__pycache__array, dim = (input_len, len(Symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t, qextSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probabilities

        """
        S, T = len(extended_symbols), len(logits)
        alpha = np.zeros(shape=(T, S))

        # -------------------------------------------->
        # TODO: Intialize alpha[0][0]
        # TODO: Intialize alpha[0][1]
        # TODO: Compute all values for alpha[t][sym] where 1 <= t < T and 1 <= sym < S (assuming zero-indexing)
        # IMP: Remember to check for skipConnect when calculating alpha
        # <---------------------------------------------
        #The forward recursion
        # First, at t = 1
        alpha[0][0] = logits[0][extended_symbols[0]]
        alpha[0][1] = logits[0][extended_symbols[1]]
        for i in range(2,S):
            alpha[0][i] = 0
        for t in range(1,T):
            alpha[t][0] = alpha[t-1][0]*logits[t][extended_symbols[0]]
            for i in range(1,S):
                alpha[t][i] = alpha[t-1][i] + alpha[t-1][i-1]
                if (skip_connect[i] == 1): # if skip connect is True: add another propability from skip
                    alpha[t][i] += alpha[t-1][i-2]
                alpha[t][i] *= logits[t][extended_symbols[i]]
        
         

        return alpha
        #raise NotImplementedError

    def get_backward_probs(self, logits, extended_symbols, skip_connect):
        """Compute backward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t,extSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probabilities
        
        """

        S, T = len(extended_symbols), len(logits)
        beta = np.zeros(shape=(T, S))
        betahat = np.zeros(shape=(T, S))

        # -------------------------------------------->
        # TODO
        # <--------------------------------------------
        
        #[Sext] = extendedsequencewithblanks(S)
        #The backward recursion
        # First, at t = T
        betahat[T-1][S-1] = logits[T-1][extended_symbols[S-1]]
        betahat[T-1][S-2] = logits[T-1][extended_symbols[S-2]]
        for i in range(1, S-2):
            betahat[T-1][i] = 0
        for t in range(T-2, -1, -1):
            betahat[t][S-1] = betahat[t+1][S-1]*logits[t][extended_symbols[S-1]]
            for i in range(S-2, -1, -1):
                betahat[t][i] = betahat[t+1][i] + betahat[t+1][i+1]
                if (i<= S-3) and (extended_symbols[i] != extended_symbols[i+2]):
                    betahat[t][i] += betahat[t+1][i+2]
                betahat[t][i] *= logits[t][extended_symbols[i]]
        #Compute beta from betahat
        for t in range(T-1, -1, -1):
            for i in range(S-1,-1,-1):
                beta[t][i] = betahat[t][i]/logits[t][extended_symbols[i]]
        

        return beta
        #raise NotImplementedError
        
    def get_posterior_probs(self, alpha, beta):
        """Compute posterior probabilities.

        Input
        -----
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probability

        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probability

        Return
        ------
        gamma: (np.array, dim = (input_len, 2 * target_len + 1))
                posterior probability

        """

        [T, S] = alpha.shape
        gamma = np.zeros(shape=(T, S))
        sumgamma = np.zeros((T,))

        # -------------------------------------------->
        # TODO
        # <---------------------------------------------
        for t in range(T):
            for i in range(S):
                gamma[t][i] = alpha[t][i] * beta[t][i]
                sumgamma[t] += gamma[t][i]
            for i in range(S):
                gamma[t][i] = gamma[t][i] / sumgamma[t]
        return gamma
        #raise NotImplementedError


class CTCLoss(object):

    def __init__(self, BLANK=0):
        """

        Initialize instance variables

        Argument(s)
        -----------
        BLANK (int, optional): blank label index. Default 0.
        
        """
        # -------------------------------------------->
        # No need to modify
        super(CTCLoss, self).__init__()
        self.BLANK = BLANK
        self.gammas = []
        self.ctc = CTC()
        # <---------------------------------------------

    def __call__(self, logits, target, input_lengths, target_lengths):

        # No need to modify
        return self.forward(logits, target, input_lengths, target_lengths)

    def forward(self, logits, target, input_lengths, target_lengths):
        """CTC loss forward

        Computes the CTC Loss by calculating forward, backward, and
        posterior proabilites, and then calculating the avg. loss between
        targets and predicted log probabilities

        Input
        -----
        logits [np.array, dim=(seq_length, batch_size, len(symbols)]:
            log probabilities (output sequence) from the RNN/GRU

        target [np.array, dim=(batch_size, padded_target_len)]:
            target sequences

        input_lengths [np.array, dim=(batch_size,)]:
            lengths of the inputs

        target_lengths [np.array, dim=(batch_size,)]:
            lengths of the target

        Returns
        -------
        loss [float]:
            avg. divergence between the posterior probability and the target

        """

        # No need to modify
        self.logits = logits
        self.target = target
        self.input_lengths = input_lengths
        self.target_lengths = target_lengths
         #####  IMP:
        #####  Output losses will be divided by the target lengths
        #####  and then the mean over the batch is taken

        # No need to modify
        B, _ = target.shape
        total_loss = np.zeros(B)
        self.extended_symbols = []

        for batch_itr in range(B):
            # -------------------------------------------->
            # Computing CTC Loss for single batch
            # Process:
            #     Truncate the target to target length
            #     Truncate the logits to input length
            #     Extend target sequence with blank
            #     Compute forward probabilities
            #     Compute backward probabilities
            #     Compute posteriors using total probability function
            #     Compute expected divergence for each batch and store it in totalLoss
            #     Take an average over all batches and return final result
            # <---------------------------------------------

            # -------------------------------------------->
            # TODO
            # <---------------------------------------------
            target_len = target_lengths[batch_itr]
            target_seq = target[batch_itr, :target_len]

            input_len = input_lengths[batch_itr]
            logits_batch = logits[ :input_len, batch_itr, :]

            extended_seq, skip_conn = self.ctc.extend_target_with_blank(target_seq)

            foward_prob = self.ctc.get_forward_probs(logits_batch, extended_seq, skip_conn)
            backward_prob = self.ctc.get_backward_probs(logits_batch, extended_seq, skip_conn)

            posterior_prob = self.ctc.get_posterior_probs(foward_prob, backward_prob)

            self.gammas.append(posterior_prob)

            T = posterior_prob.shape[0]
            S = posterior_prob.shape[1]

            for t in range(T):
                for s in range(S):
                    total_loss[batch_itr] += -(posterior_prob[t, s] * np.log(logits_batch[t, extended_seq[s]]))
            # <---------------------------------------------

        total_loss = np.sum(total_loss) / B
        
        return total_loss
        #raise NotImplementedError
        

    def backward(self):
        """
        
        CTC loss backard

        Calculate the gradients w.r.t the parameters and return the derivative
        w.r.t the inputs, xt and ht, to the cell.

        Input
        -----
        logits [np.array, dim=(seqlength, batch_size, len(Symbols)]:
            log probabilities (output sequence) from the RNN/GRU"""
        # No need to modify
        T, B, C = self.logits.shape
        dY = np.full_like(self.logits, 0)

        for batch_itr in range(B):
            # -------------------------------------------->
            # Computing CTC Derivative for single batch
            # Process:
            #     Truncate the target to target length
            #     Truncate the logits to input length
            #     Extend target sequence with blank
            #     Compute derivative of divergence and store them in dY
            # <---------------------------------------------

            # -------------------------------------------->
            # TODO
            # <---------------------------------------------
            target_len = self.target_lengths[batch_itr]
            target_seq = self.target[batch_itr, :target_len]

            input_len = self.input_lengths[batch_itr]
            logits_batch = self.logits[ :input_len, batch_itr, :]

            extended_seq, skip_conn = self.ctc.extend_target_with_blank(target_seq)
            posterior_batch = self.gammas[batch_itr]

            T = posterior_batch.shape[0]
            S = posterior_batch.shape[1]

            for t in range(T):
                for s in range(S):
                    dY[t, batch_itr, extended_seq[s]] -= posterior_batch[t, s]/(logits_batch[t,extended_seq[s]])

        return dY
        #raise NotImplementedError

