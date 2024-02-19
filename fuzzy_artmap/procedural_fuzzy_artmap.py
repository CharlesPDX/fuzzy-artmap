import numpy as np

class FuzzyArtMap:
    def __init__(self, f1_size: int = 10, f2_size: int = 10, number_of_categories: int = 2, rho_a_bar = 0):
        self.alpha = 0.001  # "Choice" parameter > 0. Set small for the conservative limit (Fuzzy AM paper, Sect.3)
        self.beta = 1  # Learning rate. Set to 1 for fast learning
        self.rho_a_bar = rho_a_bar  # Baseline vigilance for ARTa, in range [0,1]
        # use f1_size instead
        #self.M = size(a,1)  # Number of input components. Derived from data
                            # NB: Total input size = 2M (due to complement)
        # use f2_size instead
        # self.N = 20         # Number of available coding nodes
                            # We start with some resonably large number
                            # then, if we need to, can add more uncommitted
        # self.L = size(bmat,1)       # Number of output nodes. Derived from data ??? number output classes ????
        self.rho_ab = 0.95          # Map field vigilance, in [0,1]
        self.epsilon = 0.001        # Fab mismatch raises ARTa vigilance to this
                                    # much above what is needed to reset ARTa
        self.weight_a = np.ones((f2_size, f1_size)) # Initial weights in ARTa. All set to 1 Row-i, col-j entry = weight from input node i to F2 coding node j
        self.weight_ab = np.ones((f2_size, number_of_categories))  # Row-k, col-j entry = weight from ARTa F2  node j to Map Field node k
        self.committed_nodes = []
        
    def _resonance_search(self, input_vector, already_reset_nodes, rho_a, allow_category_growth = True):
        resonant_a = False
        while not resonant_a:
            # In search of a resonating ARTa node...
            # Find the winning, matching ARTa node

            N = self.weight_a.shape[0]
            # Count how many F2a nodes we have

            A_for_each_F2_node = input_vector * np.ones((N,1))
            # Matrix containing a copy of A for each F2 node. Useful for Matlab

            A_AND_w = np.minimum(A_for_each_F2_node, self.weight_a)
            # Fuzzy AND = min

            S = np.sum(A_AND_w, axis=1) # might be wrong operator
            # Row vector of signals to F2 nodes

            T = S / (self.alpha + np.sum(self.weight_a, axis=1))
            # Choice function vector for F2

            # Set all the reset nodes to zero
            T[already_reset_nodes] = np.zeros((len(already_reset_nodes), ), dtype=np.float32)

            # Finding the winning node, J

            J = np.argmax(T)
            # Matlab function max works such that J is the lowest index of max T elements, as
            # desired. J is the winning F2 category node

            # y = np.zeros((N, 1))
            # y[J]=1
            # Activities of F2. All zero, except J

            w_J = self.weight_a[J, :]  # ?????
            # Weight vector into winning F2 node, J

            x = np.minimum(input_vector, w_J)
            # Fuzzy version of 2/3 rule. x is F1 activity
            # NB: We could also use J-th element of S
            # since the top line of the match fraction
            # |I and w|/|I| is sum(x), which is
            # S = sum(A_AND_w) from above

            #####################################
            ######## Testing if the winning node resonates in ARTa

            if np.sum(x)/np.sum(input_vector) >= rho_a:
                # If a match, we're done
                resonant_a = True         # ARTa resonates
                # The while resonant_a == 0 command will stop looping
                # now, so we exit the while loop and go onto to Fab
            else:
                # If mismatch then we reset
                resonant_a = False     # So, still not resonating
                already_reset_nodes.append(J)
                # Record that node J has been reset already.

            #####################################
            # Creating a new node if we've reset all of them

            if len(already_reset_nodes) == N:
                if allow_category_growth:
                    # If all F2a nodes reset
                    self.weight_a = np.vstack((self.weight_a, np.ones((1, self.weight_a.shape[1]))))
                    self.weight_ab = np.vstack((self.weight_ab, np.ones((1, self.weight_ab.shape[1]))))
                else:
                    return -1, None
            # Give the new F2a node a w_ab entry
            # Now go back and this new node should win
        return J, x

    def train(self, input_vector: np.array, class_vector: np.array):
        rho_a = self.rho_a_bar
        # We start off with ARTa vigilance at baseline
        
        resonant_ab = False
        # Not resonating in the Fab match layer either

        already_reset_nodes = []  # We haven't rest any ARTa nodes for this input pattern yet

        while not resonant_ab:
            J, x = self._resonance_search(input_vector, already_reset_nodes, rho_a)

            # Desired output for input number i
            z = np.minimum(class_vector, self.weight_ab[J, :])   # Fab activation vector, z
            # (Called x_ab in Fuzzy ARTMAP paper)
            # Test for Fab resonance

            if np.sum(z)/np.sum(class_vector) >= self.rho_ab:     # We have an Fab match
                resonant_ab = True
            # This will cause us to leave the
            # while resonant_ab==0 loop and
            # go on to do learning.

            else: # We have an Fab mismatch
                resonant_ab = False
                # This makes us go through
                # the resonant_ab==0 loop again
                # resonant_a = False
                # This makes us go through
                # ARTa search again, this
                # search being inside the
                # resonant_ab==0 loop
                # Increase rho_a vigilance.
                # This will cause F2a node J to get reset when
                # we go back through the ARTa search loop again.
                # Also, *for this input*, the above-baseline
                # vigilance will cause a finer ARTa category to win

                rho_a = np.sum(x)/np.sum(input_vector) + self.epsilon

            # End of the while loop searching for ARTa resonance
            # If resonant_a = 0, we pick the next highest Tj
            # and see if *that* node resonates, i.e. goto "while"
            # If resonant_a = 1, we have found an ARTa resonance,
            # namely node J
            # So we go on to see if we get Fab match with node J

        #### End of the while resonant_ab==0 loop.
        #### Now we have a resonating ARTa output
        #### which gives a match at the Fab layer.
        #### So, we go on to have learning
        #### in the w_a and w_ab weights


        #### Let the winning, matching node J learn

        self.weight_a[J, :] = self.beta * x + (1-self.beta) * self.weight_a[J, :]
        # NB: x = min(A,w_J) = I and w
        #### Learning on F1a <--> f2a weights

        self.weight_ab[J, :] = self.beta * z + (1-self.beta) * self.weight_ab[J, :]
        # NB: z=min(b,w_ab(J))=b and w

    def predict(self, input_vector: np.array):
        rho_a = 0
        # We start off with ARTa vigilance at baseline
        resonant_a = False

        # We're not resonating in the ARTa module yet
        resonant_ab = False

        # Not resonating in the Fab match layer either
        already_reset_nodes = []  # We haven't rest any ARTa nodes for this input pattern yet

        while not resonant_ab:
            J, x = self._resonance_search(input_vector, already_reset_nodes, rho_a, False)

            # Desired output for input number i
            if J == -1:
                return np.zeros_like(self.weight_ab)
            
            z = self.weight_ab[J, None]   # Fab activation vector, z
            # prediction_transliteration = self.weight_ab[:,J]/sum(self.weight_ab[:,J])
            # (Called x_ab in Fuzzy ARTMAP paper)
            resonant_ab = True
            # prediction_transliteration = self.weight_ab[J,:]/np.sum(self.weight_ab[J,:])
            # print(prediction_transliteration)


            # End of the while loop searching for ARTa resonance
            # If resonant_a = 0, we pick the next highest Tj
            # and see if *that* node resonates, i.e. goto "while"
            # If resonant_a = 1, we have found an ARTa resonance,
            # namely node J
            # So we go on to see if we get Fab match with node J

        #### End of the while resonant_ab==0 loop.
        #### Now we have a resonating ARTa output
        #### which gives a match at the Fab layer.
        #### So, we go on to have learning
        #### in the w_a and w_ab weights


        #### Let the winning, matching node J learn

        # self.weight_a[:, J, np.newaxis] = self.beta * x + (1-self.beta) * self.weight_a[:, J, np.newaxis]
        # # NB: x = min(A,w_J) = I and w
        # #### Learning on F1a <--> f2a weights

        # self.weight_ab[J, :, np.newaxis] = self.beta * z + (1-self.beta) * self.weight_ab[J, :, np.newaxis]
        # NB: z=min(b,w_ab(J))=b and w
        return z
    
    @staticmethod
    def complement_encode(original_vector: np.array) -> np.array:
        complement = 1-original_vector
        complement_encoded_value = np.hstack((original_vector,complement))
        return complement_encoded_value