import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import SGD
from tqdm import tqdm
from source.utility import get_time
from source.model.MultiFTNet import MultiFTNet
from source.data_io.dataset_loader import get_train_loader

class TrainMain:
    def __init__(self,conf):
        self.conf = conf
        self.board_loss_every = conf.board_loss_every
        self.save_energy=conf.save_energy
        self.step=0
        self.start_epoch=0
        self.train_loader=get_train_loader(self.conf)


    def train_model(self):
        self.__init__model_param()
        self._train_stage()

    def __init__model_param(self):
        self.cls_criterion=tf.keras.losses.CategoricalCrossEntrophy(from_logits=True)
        self.ft_criterion= tf.keras.losses.MeanSquaredError()
        self.model=self.__define_network()
        self.optimizer= tf.keras.optimizers.SGD(
            learning_rate=self.conf.lr,
            weight_decay=5e-4,
            momentum=self.conf.momentum
        )
        
        steps_per_epoch = len(train_dataset)

        self.schedule_lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=[m * steps_per_epoch for m in self.conf.milestones],
            values=[self.conf.lr * (self.conf.gamma ** i)
                    for i in range(len(self.conf.milestones) + 1)]
        )

        print("lr: ", self.conf.lr)
        print("epochs: ", self.conf.epochs)
        print("milestones: ", self.conf.milestones)

    def _train_stage(self):
        # self.model.train()
        running_loss = 0.
        running_acc = 0.
        running_loss_cls = 0.
        running_loss_ft = 0.

        is_first = True
        for e in range(self.start_epoch, self.conf.epochs):
            if is_first:
                self.writer = tf.summary.create_file_writer(self.conf.log_path)
                is_first = False
            print('epoch {} started'.format(e))
            print("lr: ", self.schedule_lr.get_lr())

            for sample, ft_sample, target in tqdm(iter(self.train_loader)):
                imgs = [sample, ft_sample]
                labels = target

                loss, acc, loss_cls, loss_ft = self._train_batch_data(imgs, labels)
                running_loss_cls += loss_cls
                running_loss_ft += loss_ft
                running_loss += loss
                running_acc += acc

                self.step += 1

                if self.step % self.board_loss_every == 0 and self.step != 0:
                    loss_board = running_loss / self.board_loss_every
                    acc_board = running_acc / self.board_loss_every
                    loss_cls_board = running_loss_cls / self.board_loss_every
                    loss_ft_board = running_loss_ft / self.board_loss_every

    # LR hiện tại (nếu optimizer dùng schedule thì cái này vẫn đúng)
                    lr = self.optimizer.learning_rate(self.optimizer.iterations)

                    with self.writer.as_default():
                        tf.summary.scalar('Training/Loss', loss_board, step=self.step)
                        tf.summary.scalar('Training/Acc', acc_board, step=self.step)
                        tf.summary.scalar('Training/Learning_rate', lr, step=self.step)
                        tf.summary.scalar('Training/Loss_cls', loss_cls_board, step=self.step)
                        tf.summary.scalar('Training/Loss_ft', loss_ft_board, step=self.step)
                    

                    running_loss = 0.
                    running_acc = 0.
                    running_loss_cls = 0.
                    running_loss_ft = 0.
                if self.step % self.save_every == 0 and self.step != 0:
                    time_stamp = get_time()
                    self._save_state(time_stamp, extra=self.conf.job_name)
            self.schedule_lr.step()

        time_stamp = get_time()
        self._save_state(time_stamp, extra=self.conf.job_name)
        self.writer.close()

    def _train_batch_data(self, imgs, labels):
        sample, ft_sample = imgs  # imgs = [sample, ft_sample]

        with tf.GradientTape() as tape:
            embeddings, feature_map = self.model(sample, training=True)

            # chú ý: TF loss nhận (y_true, y_pred)
            loss_cls = self.cls_criterion(labels, embeddings)
            loss_fea = self.ft_criterion(ft_sample, feature_map)

            loss = 0.5 * loss_cls + 0.5 * loss_fea

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        preds = tf.argmax(embeddings, axis=1, output_type=labels.dtype)
        acc = tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))

        return float(loss.numpy()), float(acc.numpy()), float(loss_cls.numpy()), float(loss_fea.numpy())
    
    def _define_network(self):
        param = {
            'num_classes': self.conf.num_classes,
            'img_channel': self.conf.input_channel,
            'embedding_size': self.conf.embedding_size,
            'conv6_kernel': self.conf.kernel_size
        }

        # MultiFTNet PHẢI là tf.keras.Model
        model = MultiFTNet(**param)
        return model
    

    def _get_accuracy(self, output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))


        ret = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(dim=0, keepdim=True)
            ret.append(correct_k.mul_(1. / batch_size))
        return ret

    def _save_state(self, time_stamp, extra=None):
        save_path = self.conf.model_path
        filename = '{}_{}_model_iter-{}'.format(time_stamp, extra, self.step)

        # đảm bảo thư mục tồn tại
        tf.io.gfile.makedirs(save_path)

        # lưu weights (tương đương state_dict của PyTorch)
        self.model.save_weights(
            save_path + '/' + filename + '.weights.h5'
        )



