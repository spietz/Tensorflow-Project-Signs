from base.base_train import BaseTrain
# from tqdm import tqdm
import numpy as np


class SignsTrainer(BaseTrain):
    def __init__(self, sess, model, data, config,logger):
        super(SignsTrainer, self).__init__(sess, model, data, config,logger)

    def train_epoch(self):  # in each epoch all minibatches are looped once (oder was?)
        # loop = tqdm(range(self.config.num_iter_per_epoch))  # enables progress for the loop
        loop = range(self.config.num_iter_per_epoch)  # enables progress for the loop
        losses = []
        accs = []
        for _ in loop:
            loss, acc = self.train_step()  # train 1 one minibatch
            losses.append(loss)
            accs.append(acc)
        loss = np.mean(losses)  # average loss of minibatches
        acc = np.mean(accs)  # average accuracry of minibatches

        cur_it = self.model.global_step_tensor.eval(self.sess)

        # logging
        self.logger.summarize(cur_it, summaries_dict={'loss': loss, 'acc': acc})

        # # checkpoint, how often?
        # if cur_it % self.config.save_interval == 0:
        #     self.model.save(self.sess)

    def train_step(self):

        minibatch_x, minibatch_y = next(self.data.next_batch(self.config.batch_size))

        feed_dict = {self.model.x: minibatch_x,
                     self.model.y: minibatch_y,
                     self.model.is_training: True}

        _, loss, acc = self.sess.run(fetches=[self.model.train_step,  # adam-optimizer
                                              self.model.cross_entropy,  # cost function
                                              self.model.accuracy],  # accuracy
                                     feed_dict=feed_dict)

        return loss, acc
