import tensorflow as tf
import numpy as np

def unconditional_model_build(random_seed = 1):

    timesteps = 1200                     # LSTM network is unrolled for 1200 timesteps
    hidden_dim = 30 #300
    epochs = 10
    batch_size = 10
    text_len = 70                                # all the text inputs are expanded to this length
    char_dims = 78 # len(char_encoding)
    num_gaussians_mixturemodel = 20                # number of gaussians in the mixture model to predict outputs
    num_gaussians_windowmm = 10                  # number of gaussians in the mixture model to predict 'w'
    output_dim = 6*num_gaussians_mixturemodel + 1  # =121
    bias = 0
	
    g1 = tf.Graph() #
    with g1.as_default():
        with tf.variable_scope("Handwriting_Prediction") :
            #  Sequences we will provide at runtime
            seq_len = tf.placeholder(tf.int32, [batch_size]) ######
            stroke_input = tf.placeholder(tf.float32, [batch_size, timesteps, 3])  
            initializer = tf.contrib.layers.xavier_initializer()

            stroke_list = tf.transpose(stroke_input, perm = [1,0,2])
            inputs = tf.unstack(value = stroke_list, axis = 0)
            with tf.variable_scope("Recurrent_Connections") :
                cell1 = tf.contrib.rnn.LSTMCell(hidden_dim, 3, initializer=initializer)
                initial_state1 = cell1.zero_state(batch_size, tf.float32)
                outputs1, states1 = tf.contrib.rnn.static_rnn(cell1, inputs, initial_state=initial_state1, scope="RNN1", sequence_length=seq_len)

                input_layer2 = tf.unstack(tf.concat([outputs1, inputs], 2), axis = 0)        # uses skip connections to easen gradient flow -> inputs2 = outputs1+inputs1

                cell2 = tf.contrib.rnn.LSTMCell(hidden_dim, hidden_dim + 3, initializer=initializer)
                initial_state2 = cell2.zero_state(batch_size, tf.float32)
                outputs2, states2 = tf.contrib.rnn.static_rnn(cell2, input_layer2, initial_state=initial_state2, scope="RNN2", sequence_length=seq_len)

                input_layer3 = tf.unstack(tf.concat([outputs2, inputs], 2), axis = 0)       # inputs3 = outputs3+inputs1

                cell3 = tf.contrib.rnn.LSTMCell(hidden_dim, hidden_dim+3, initializer=initializer)
                initial_state3 = cell3.zero_state(batch_size, tf.float32)
                outputs3, states3 = tf.contrib.rnn.static_rnn(cell3, input_layer3, initial_state=initial_state3,scope="RNN3", sequence_length=seq_len)

            with tf.variable_scope("Dense_Connections") :
                input_layer4 = tf.reshape(tf.transpose(tf.concat([outputs3, outputs2, outputs1], 2), perm = [1,0,2]), [batch_size*timesteps, 3*hidden_dim])
                # inputs to the dense layer are the hidden units of all the three RNNs
                output = tf.layers.dense(input_layer4, output_dim)
                output_list = tf.reshape(output, [batch_size, timesteps, output_dim])

                x3 = 1 / (1 + tf.exp(output_list[:, :, 0]))
                weights, x1_mean, x2_mean, x1_var, x2_var, correlation = tf.split(axis = 2, num_or_size_splits = 6, value = output_list[:, :, 1:])      
                weight_norm = tf.exp(weights * (1 + bias)) / tf.tile(tf.reduce_sum(tf.exp(weights * (1 + bias)), 2, keep_dims = True), [1,1,num_gaussians_mixturemodel])
                x1_var_norm = tf.exp(x1_var - bias)
                x2_var_norm = tf.exp(x2_var - bias)
                correlation_norm = tf.tanh(correlation)

                x3_target, x2_target, x1_target = tf.unstack(tf.concat([stroke_input[:, 1:, :],[[[1,0,0]]]*batch_size], axis = 1), axis = 2)
                x1_target_M = tf.reshape(tf.tile(x1_target, [1,num_gaussians_mixturemodel]), [batch_size, timesteps, num_gaussians_mixturemodel])
                x2_target_M = tf.reshape(tf.tile(x2_target, [1,num_gaussians_mixturemodel]), [batch_size, timesteps, num_gaussians_mixturemodel])

                z = tf.square((x1_target_M-x1_mean)/x1_var_norm) + tf.square((x2_target_M-x2_mean)/x2_var_norm) - 2*correlation_norm*(x1_target_M-x1_mean)*(x2_target_M-x2_mean)/(x1_var_norm*x2_var_norm)
                denominator = 2*np.pi*x1_var_norm*x2_var_norm*tf.sqrt(1 - tf.square(correlation_norm))

                Prob = tf.exp(-z/(2*(1-tf.square(correlation_norm)))) / denominator
                Prob_norm = Prob*weight_norm
                prediction_loss = -1*tf.reduce_mean(tf.reduce_sum(tf.log(tf.reduce_sum(Prob_norm, axis = 2)), axis=1))/100
                cross_entropy_loss = -1*tf.reduce_mean(tf.reduce_sum(x3_target*tf.log(x3)+(1-x3_target)*tf.log(1-x3), axis = 1))
                loss = prediction_loss + cross_entropy_loss

            #train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

            optimizer = tf.train.AdamOptimizer(1e-4)
            gradients_variables = optimizer.compute_gradients(loss)
            recurrent_gvs = [(tf.clip_by_value(grad, -10, 10), var) for grad, var in gradients_variables if var.name.startswith("Handwriting_Prediction/Recurrent_Connections")]
            dense_gvs = [(tf.clip_by_value(grad, -100, 100), var) for grad, var in gradients_variables if var.name.startswith("Handwriting_Prediction/Dense_Connections")]
            # gradient clipping to avoid exploding gradient problem
            recurrent_vars = [var for grad, var in gradients_variables if var.name.startswith("Handwriting_Prediction/Recurrent_Connections")]
            dense_vars = [var for grad, var in gradients_variables if var.name.startswith("Handwriting_Prediction/Dense_Connections")]

            other_gvs = [(grad, var) for grad, var in gradients_variables if var not in (recurrent_vars+dense_vars)]
            train_step = optimizer.apply_gradients(recurrent_gvs+dense_gvs+other_gvs)
            #gradients, _ = tf.clip_by_global_norm(gradients, 100.0) 
            # use of tf.clip_by_global_norm allows better convergence as it doesn't introduce bias/avoids spurious gradient directions

        # Create initialize op, this needs to be run by the session!
        iop = tf.initialize_all_variables()
        saver = tf.train.Saver()
        print "done"
        #tf.get_variable_scope().reuse_variables() 

    session = tf.Session(graph = g1)
    session.run(iop)
    #save_path = saver.restore(session, "Checkpoints/Unconditional_model.ckpt")
    
    initial = np.zeros([3,], np.float32)
    np.random.seed(random_seed)
    initial[0] = 1
    initial[1:3] = np.random.rand(2)
    strokes_test = np.zeros([batch_size, timesteps, 3], dtype=np.float32)
    stroke_output = np.zeros([timesteps, 3], dtype=np.float32)
    state1, state2, state3 = session.run([initial_state1,initial_state2,initial_state3])
    stroke_output[0, :] = initial[:]
    for i in range(timesteps-1):
        strokes_test[0, 0, :] = initial[:]
        feed_dict = {stroke_input: strokes_test, seq_len: [1]*10, initial_state1: state1, initial_state2: state2, initial_state3: state3}  
        stroke_prob, prob_weights, mu1, mu2, var1, var2, corel_coeff, state1, state2, state3 = session.run([x3, weight_norm, x1_mean, x2_mean,x1_var_norm, x2_var_norm, correlation_norm, states1, states2, states3], feed_dict=feed_dict)

        stroke_prob = stroke_prob[0,0]
        prob_weights = prob_weights[0,0,:]
        means_matrix = np.zeros([num_gaussians_mixturemodel, 2], np.float32)
        means_matrix[:,0] = mu1[0,0,:]
        means_matrix[:,1] = mu2[0,0,:]
        cov_matrix = np.zeros([num_gaussians_mixturemodel, 2, 2], np.float32)
        for it in range(num_gaussians_mixturemodel):
            cov_matrix[it,:,:] = [[var1[0,0,it]*var1[0,0,it], corel_coeff[0,0,it]*var1[0,0,it]*var2[0,0,it]],[corel_coeff[0,0,it]*var1[0,0,it]*var2[0,0,it], var2[0,0,it]*var2[0,0,it]]]

      #########################################
        temp = np.array([0,0,0], np.float32)
        r = np.random.rand()
        if r < stroke_prob :
            temp[0] = 1
        for m in range(num_gaussians_mixturemodel):
            temp[1:3] += prob_weights[m]*np.random.multivariate_normal(means_matrix[m], cov_matrix[m])
      #########################################

        stroke_output[i+1,:] = temp[:]
    return stroke_output

def conditional_model_build(text_input='welcome to lyrebird', random_seed=1):

    timesteps = 1200                     # LSTM network is unrolled for 1200 timesteps
    hidden_dim = 30 #300
    epochs = 10
    batch_size = 10
    text_len = 70                                # all the text inputs are expanded to this length
    char_dims = 78 # len(char_encoding)
    num_gaussians_mixturemodel = 20                # number of gaussians in the mixture model to predict outputs
    num_gaussians_windowmm = 10                  # number of gaussians in the mixture model to predict 'w'
    output_dim = 6*num_gaussians_mixturemodel + 1  # =121
    bias = 0

    g1 = tf.Graph()
    with g1.as_default():
        with tf.variable_scope("Handwriting_Synthesis") :    

            initializer = tf.contrib.layers.xavier_initializer()

            stroke_input = tf.placeholder(tf.float32, [batch_size, timesteps, 3])
            stroke_list = tf.transpose(stroke_input, perm = [1,0,2])               # inputs for an RNN must be a list of
            inputs = tf.unstack(value = stroke_list, axis = 0)                     # T timesteps with units of size [batch_size,input_dims]
            seq_len = tf.placeholder(tf.int32, [batch_size])                       # used to dynamically vary the sequence length while training

            loss_scale = tf.placeholder(tf.float32, [2])                           # used to adjust the scales of both the losses to improve training

            #text_len = tf.placeholder(tf.int32, [batch_size])    
            text_conditioned = tf.placeholder(tf.float32, [batch_size, text_len, char_dims])    


            w_list_prev = tf.zeros([batch_size, char_dims], tf.float32)                    # stores inputs from previous timesteps
            kappa_prev = tf.zeros([batch_size, num_gaussians_windowmm,1], tf.float32)      # stores inputs from previous timesteps

            w_list_values = []                                                             # stores a list of all the w's across timesteps
            kappa_values = []                                                              # stores a list of all the kappa's across timesteps

            u = tf.cast(tf.tile(tf.expand_dims(tf.expand_dims(tf.range(text_len), 0), 0), [batch_size, num_gaussians_windowmm, 1]), tf.float32)            # shape [batch_size,num_gaussians_windowmm,70]

            cell1 = tf.contrib.rnn.LSTMCell(hidden_dim, 3+char_dims, initializer=initializer)
            initial_state1 = cell1.zero_state(batch_size, tf.float32)

            intermediate_w = tf.get_variable(name = "intermediate_weight", dtype=tf.float32, shape=[hidden_dim, 3*num_gaussians_windowmm])                 # used for computing (alpha,beta,kappa) -> 'w'
            intermediate_b = tf.get_variable(name = "intermediate_bias", dtype=tf.float32, shape=[3*num_gaussians_windowmm])

            cell2 = tf.contrib.rnn.LSTMCell(hidden_dim, hidden_dim+char_dims+3, initializer=initializer)
            initial_state2 = cell2.zero_state(batch_size, tf.float32)

            output_w = tf.get_variable(name = "output_weight", dtype=tf.float32, shape=[hidden_dim, output_dim])                                           # used to compute the outputs (e,[pi,mu1,mu2,sigma1,
            output_b = tf.get_variable(name = "output_bias", dtype=tf.float32, shape=[output_dim])                                                         # sigma2,correlation coefficient]*{#gaussians})

            output_list = []                                                                                                                               # stores a list of all the outputs

        for t in range(timesteps):                            # unrolling the LSTM network for 1200 timesteps
            with tf.variable_scope("Recurrent_Connections") :  
                input_layer1 = [tf.concat([w_list_prev, inputs[t]], 1)]
                outputs1, states1 = tf.contrib.rnn.static_rnn(cell1, input_layer1, initial_state=initial_state1, scope="RNN1")

                alpha_hat, beta_hat, kappa_hat = tf.split(tf.reshape(tf.nn.xw_plus_b(outputs1[0], intermediate_w, intermediate_b),[batch_size,3*num_gaussians_windowmm]), 3, 1)
                alpha_norm = tf.expand_dims(tf.exp(alpha_hat), 2)                                                                                          # (alpha,beta,kappa) each have a shape
                beta_norm = tf.expand_dims(tf.exp(beta_hat), 2)                                                                                            # of [batch_size,num_gaussians_windowmm, 1]
                kappa_norm = kappa_prev + tf.expand_dims(tf.exp(kappa_hat), 2)
                #####################################################################
                phi = tf.reduce_sum(tf.exp(tf.square(-u + kappa_norm) * (-beta_norm)) * alpha_norm, 1, keep_dims=True)                                     # phi shape [batch_size,1,70]           
                w_list = tf.squeeze(tf.matmul(phi, text_conditioned), [1])                                                                                 # text_conditioned shape [batch_size,70,78]
                w_list_values.append(w_list)                                                                                                               # w_list shape [batch_size,78]
                kappa_values.append(kappa_norm)
                #####################################################################
                kappa_prev = kappa_norm
                w_list_prev = w_list

                input_layer2 = [tf.concat([w_list, outputs1[0], inputs[t]], 1)]
                outputs2, states2 = tf.contrib.rnn.static_rnn(cell2, input_layer2, initial_state=initial_state2, scope="RNN2")

                # inputs -> RNN1 -> Dense1 -> (alpha,beta,kappa) -> w -> RNN2 -> Dense2 -> (e,[pi,mu1,mu2,sigma1,sigma2,correlation coefficient]*{#gaussians}) 

            with tf.variable_scope("Dense_Connections") :
                output_list.append(tf.reshape(tf.nn.xw_plus_b(outputs2[0], output_w, output_b),[batch_size, output_dim]))

        with tf.variable_scope("Dense_Connections") :
            final_output = tf.transpose(tf.stack(output_list,0), perm = [1,0,2])
            list_of_w = tf.stack(w_list_values,0)
            list_of_kappa = tf.stack(kappa_values,0)

            x3 = 1 / (1 + tf.exp(final_output[:,:,0]))
            #print x3
            weights, x1_mean, x2_mean, x1_var, x2_var, correlation = tf.split(axis = 2, num_or_size_splits = 6, value = final_output[:, :, 1:])
            weight_norm = tf.exp(weights * (1 + bias)) / tf.tile(tf.reduce_sum(tf.exp(weights * (1 + bias)), 2, keep_dims = True), [1,1,num_gaussians_mixturemodel])       # parameters of predicted mixture model
            #print x1_mean
            x1_var_norm = tf.exp(x1_var - bias)
            x2_var_norm = tf.exp(x2_var - bias)
            correlation_norm = tf.tanh(correlation)

            x3_target, x2_target, x1_target = tf.unstack(tf.concat([stroke_input[:, 1:, :],[[[1,0,0]]]*batch_size], axis = 1), axis = 2)
            x1_target_M = tf.reshape(tf.tile(x1_target, [1,num_gaussians_mixturemodel]), [batch_size, timesteps, num_gaussians_mixturemodel])          # target sequence
            x2_target_M = tf.reshape(tf.tile(x2_target, [1,num_gaussians_mixturemodel]), [batch_size, timesteps, num_gaussians_mixturemodel])

            z = tf.square((x1_target_M-x1_mean)/x1_var_norm) + tf.square((x2_target_M-x2_mean)/x2_var_norm) - 2*correlation_norm*(x1_target_M-x1_mean)*(x2_target_M-x2_mean)/(x1_var_norm*x2_var_norm)
            denominator = 2*np.pi*x1_var_norm*x2_var_norm*tf.sqrt(1 - tf.square(correlation_norm))

            Prob = tf.exp(-z/(2*(1-tf.square(correlation_norm)))) / denominator
            Prob_norm = Prob*weight_norm
            prediction_loss = -1*tf.reduce_mean(tf.reduce_sum(tf.log(tf.reduce_sum(Prob_norm, axis = 2)), axis=1))/loss_scale[0]
            cross_entropy_loss = -1*tf.reduce_mean(tf.reduce_sum(x3_target*tf.log(x3)+(1-x3_target)*tf.log(1-x3), axis = 1))/loss_scale[1]
            loss = cross_entropy_loss + prediction_loss

        #train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

        optimizer = tf.train.AdamOptimizer(1e-4)
        gradients_variables = optimizer.compute_gradients(loss)
        recurrent_gvs = [(tf.clip_by_value(grad, -10, 10), var) for grad, var in gradients_variables if var.name.startswith("Handwriting_Prediction/Recurrent_Connections")]     # Gradient clipping to avoid 
        dense_gvs = [(tf.clip_by_value(grad, -100, 100), var) for grad, var in gradients_variables if var.name.startswith("Handwriting_Prediction/Dense_Connections")]           # exploding gradeints problem

        recurrent_vars = [var for grad, var in gradients_variables if var.name.startswith("Handwriting_Synthesis/Recurrent_Connections")]
        dense_vars = [var for grad, var in gradients_variables if var.name.startswith("Handwriting_Synthesis/Dense_Connections")]

        other_gvs = [(tf.clip_by_value(grad, -10, 10), var) for grad, var in gradients_variables if var not in (recurrent_vars+dense_vars)]
        train_step = optimizer.apply_gradients(recurrent_gvs+dense_gvs+other_gvs)
        #gradients, _ = tf.clip_by_global_norm(gradients, 100.0) 
        # use of tf.clip_by_global_norm allows better convergence as it doesn't introduce bias/avoids spurious gradient directions

        # Create initialize op, this needs to be run by the session!
        iop = tf.initialize_all_variables()
        saver = tf.train.Saver()
        #tf.get_variable_scope().reuse_variables()
    
    session = tf.Session(graph = g1)
    session.run(iop)
    
    #save_path = saver.restore(session, "Checkpoints/conditional_model.ckpt")
    
    initial = np.zeros([3,], np.float32)
    np.random.seed(random_seed)
    initial[0] = 1
    initial[1:3] = np.random.rand(2)
    strokes_test = np.zeros([batch_size, timesteps, 3], dtype=np.float32)
    stroke_output = np.zeros([timesteps, 3], dtype=np.float32)
    state1, state2 = session.run([initial_state1,initial_state2])
    stroke_output[0, :] = initial[:]

    w_list_prv = np.zeros([batch_size, char_dims], np.float32)
    kappa_prv = np.zeros([batch_size, num_gaussians_windowmm,1], np.float32)

    for i in range(timesteps-1):
        strokes_test[0, 0, :] = initial[:]
        batch_c = np.zeros([batch_size,70,78], dtype=np.float32)
        batch_c[0] = text_input
        feed_dict = {stroke_input: strokes_test, seq_len: [1]*batch_size, initial_state1: state1, initial_state2: state2, text_conditioned: batch_c, kappa_prev: kappa_prv, w_list_prev: w_list_prv}  

        w_list_val, kappa_val, stroke_prob, prob_weights, mu1, mu2, var1, var2, corel_coeff, state1, state2 = session.run([list_of_w, list_of_kappa, x3, weight_norm, x1_mean, x2_mean,x1_var_norm, x2_var_norm, correlation_norm, states1, states2], feed_dict=feed_dict)

        w_list_prv = w_list_val[0]
        kappa_prv = kappa_val[0]

        stroke_prob = stroke_prob[0,0]
        prob_weights = prob_weights[0,0,:]
        means_matrix = np.zeros([num_gaussians_mixturemodel, 2], np.float32)
        means_matrix[:,0] = mu1[0,0,:]
        means_matrix[:,1] = mu2[0,0,:]
        cov_matrix = np.zeros([num_gaussians_mixturemodel, 2, 2], np.float32)
        for it in range(num_gaussians_mixturemodel):
            cov_matrix[it,:,:] = [[var1[0,0,it]*var1[0,0,it], corel_coeff[0,0,it]*var1[0,0,it]*var2[0,0,it]],[corel_coeff[0,0,it]*var1[0,0,it]*var2[0,0,it], var2[0,0,it]*var2[0,0,it]]]

      #########################################
        temp = np.array([0,0,0], np.float32)
        r = np.random.rand()
        if r < stroke_prob :
            temp[0] = 1
        #test = GaussianMixture(weights_ = prob_weights, means_ = means_matrix, covariances_ = cov_matrix).sample()
        for m in range(num_gaussians_mixturemodel):
            temp[1:3] += prob_weights[m]*np.random.multivariate_normal(means_matrix[m], cov_matrix[m])
      #########################################

        stroke_output[i+1,:] = temp[:]
    #plot_stroke(stroke_output)
    # Output:
    #   stroke - numpy 2D-array (T x 3)
    return stroke_output  

def recognition_model_build(stroke):

    timesteps = 1200                     # LSTM network is unrolled for 1200 timesteps
    hidden_dim = 30 #300
    epochs = 10
    batch_size = 10
    text_len = 70                                # all the text inputs are expanded to this length
    char_dims = 78 # len(char_encoding)
    num_gaussians_mixturemodel = 20                # number of gaussians in the mixture model to predict outputs
    num_gaussians_windowmm = 10                  # number of gaussians in the mixture model to predict 'w'
    output_dim = 6*num_gaussians_mixturemodel + 1  # =121
    bias = 0
	
    g1 = tf.Graph()
    with g1.as_default():
        with tf.variable_scope("Handwriting_Prediction") :
            seq_len = tf.placeholder(tf.int32, [batch_size])
            stroke_input = tf.placeholder(tf.float32, [batch_size, timesteps, 3])

            labels = tf.sparse_placeholder(tf.int32)
            initializer = tf.contrib.layers.xavier_initializer()

            stroke_list = tf.transpose(stroke_input, perm = [1,0,2])   

            inputs = tf.unstack(value = stroke_list, axis = 0)

            with tf.variable_scope("Recurrent_Connections") :
                cell_fwd = tf.contrib.rnn.LSTMCell(hidden_dim, 3, initializer=initializer)                          # 1 layer BiLSTM
                cell_bkd = tf.contrib.rnn.LSTMCell(hidden_dim, 3, initializer=initializer)
                initial_state_fwd = cell_fwd.zero_state(batch_size, tf.float32)
                initial_state_bkd = cell_bkd.zero_state(batch_size, tf.float32)
                outputs_fw_bw, states_fw, state_bw = tf.contrib.rnn.stack_bidirectional_rnn(cells_fw = [cell_fwd], cells_bw = [cell_bkd], inputs = inputs, initial_states_fw=[initial_state_fwd], initial_states_bw=[initial_state_bkd], scope="Stacked_BiLSTM", sequence_length=seq_len)



            with tf.variable_scope("Dense_Connections") :
                output_w = tf.get_variable(name = "output_weight", dtype=tf.float32, shape=[2*hidden_dim, output_dim])
                output_b = tf.get_variable(name = "output_bias", dtype=tf.float32, shape=[output_dim])
                output_list = tf.reshape(tf.nn.xw_plus_b( tf.reshape(outputs_fw_bw, [-1,2*hidden_dim]) , output_w, output_b), [-1,batch_size,output_dim])

                loss = tf.reduce_sum(tf.nn.ctc_loss(labels, output_list, seq_len))                 # uses CTC loss

                decoded, log_prob = tf.nn.ctc_greedy_decoder(output_list, seq_len) # tf.contrib.ctc.ctc_beam_search_decoder is slower but provides better decoding

                ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),labels))


            with tf.name_scope('training'):
                steps = tf.Variable(0.)
                learning_rate = tf.train.exponential_decay(0.001, steps, staircase=True, decay_steps=10000, decay_rate=0.5)
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                grad, var = zip(*optimizer.compute_gradients(loss))
                grad, _ = tf.clip_by_global_norm(grad, 1.)
                train_step = optimizer.apply_gradients(zip(grad, var), global_step=steps)


        iop = tf.initialize_all_variables()
        saver = tf.train.Saver()
        print "done"
    session = tf.Session(graph = g1)
    session.run(iop)
    
    #save_path = saver.restore(session, "Checkpoints/Recognition_model.ckpt")
    seq_length = [1]*batch_size
    seq_length[0] = len(stroke)
    stroke = np.vstack((stroke, [[1,0,0]]*(1200-len(stroke))))
    strokes_test = np.zeros([batch_size, timesteps, 3], dtype=np.float32)
    stroke_test[0] = stroke_modified
    
    feed = {stroke_input: strokes_test, seq_len : seq_length}
    decoded_string_num = session.run([decoded], feed_dict=feed)
    return decoded_string_num[0]
