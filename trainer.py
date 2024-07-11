import torch.nn as nn
from torch.nn.parameter import Parameter
import torch
from torch.utils import tensorboard
import utils
from model import *
import matplotlib.pyplot as plt
import pickle
from gensim.matutils import corpus2csc
from gensim.models.coherencemodel import CoherenceModel
from scipy.sparse import coo_matrix
import tqdm
from palmettopy.palmetto import Palmetto

palmetto = Palmetto("https://palmetto.demos.dice-research.org/service/")

class GBN_trainer:
    def __init__(self, args, voc_path='voc.txt'):
        self.args = args
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.save_path = args.save_path
        self.epochs = args.epochs
        self.voc = self.get_voc(voc_path)
        self.layer_num = len(args.topic_size)
        self.save_phi_every_epoch = 1000
        self.use_pal = args.use_pal
        dataset_dir = self.args.dataset_dir
        # if dataset_dir[-8:] == '20ng.pkl':
        #     with open('./dataset/20ng.pkl', 'rb') as f:
        #         dataset = pickle.load(f)
        # if dataset_dir[-6:] == 'R8.pkl':
        #     with open('./dataset/R8.pkl', 'rb') as f:
        #         dataset = pickle.load(f)
        # if dataset_dir[-8:] == 'zhdd.pkl':
        #     with open('./dataset/zhdd.pkl', 'rb') as f:
        #         dataset = pickle.load(f)
        # if dataset_dir[-9:] == 'trump.pkl':
        #     with open('./dataset/trump.pkl', 'rb') as f:
        #         dataset = pickle.load(f)
        # if dataset_dir[-15:] == 'threetopics.pkl':
        #     with open('./dataset/threetopics.pkl', 'rb') as f:
        #         dataset = pickle.load(f)
        with open(dataset_dir, 'rb') as f:
            dataset = pickle.load(f)
        labels = dataset['label']
        self.count = len(labels)

        self.model = GBN_model(args)
        self.optimizer = torch.optim.Adam([{'params': self.model.h_encoder.parameters()},
                                           {'params': self.model.shape_encoder.parameters()},
                                           {'params': self.model.scale_encoder.parameters()}],
                                          lr=self.lr, weight_decay=self.weight_decay)

        self.decoder_optimizer = torch.optim.Adam(self.model.decoder.parameters(),
                                                  lr=self.lr, weight_decay=self.weight_decay)
        
        if self.save_path != 'saves/gbn_model_weibull_etm_share_50_7_kl_0.1':
            self.log_path = os.path.join(args.save_path, "log.txt")
            utils.add_show_log(self.log_path, str(args))

    def train(self, train_data_loader):

        for epoch in range(self.epochs):

            for t in range(self.layer_num - 1):
                self.model.decoder[t + 1].rho = self.model.decoder[t].alphas

            self.model.to(self.args.device)

            loss_t = [0] * (self.layer_num + 1)
            likelihood_t = [0] * (self.layer_num + 1)
            num_data = len(train_data_loader)

            for i, (train_data, _) in enumerate(train_data_loader):
                self.model.h_encoder.train()
                self.model.shape_encoder.train()
                self.model.scale_encoder.train()
                self.model.decoder.eval()

                train_data = torch.tensor(train_data, dtype=torch.float).to(self.args.device)
                # train_label = torch.tensor(train_label, dtype=torch.long).cuda()

                re_x, theta, loss_list, likelihood = self.model(train_data)

                for t in range(self.layer_num + 1):
                    if t == 0:
                        (10.0 * loss_list[t]).backward(retain_graph=True)
                        loss_t[t] += loss_list[t].item() / num_data
                        likelihood_t[t] += likelihood[t].item() / num_data

                    elif t < self.layer_num:
                        (10.0 * loss_list[t]).backward(retain_graph=True)
                        loss_t[t] += loss_list[t].item() / num_data
                        likelihood_t[t] += likelihood[t].item() / num_data
                    else:
                        (0 * loss_list[t]).backward(retain_graph=True)
                        loss_t[t] += loss_list[t].item() / num_data
                        likelihood_t[t] += likelihood[t].item() / num_data

                for para in self.model.parameters():
                    flag = torch.sum(torch.isnan(para))

                if (flag == 0):
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=100, norm_type=2)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                self.model.h_encoder.eval()
                self.model.shape_encoder.eval()
                self.model.scale_encoder.eval()
                self.model.decoder.train()

                re_x, theta, loss_list, likelihood = self.model(train_data)

                for t in range(self.layer_num + 1):
                    if t == 0:
                        (10.0 * loss_list[t]).backward(retain_graph=True)
                        loss_t[t] += loss_list[t].item() / num_data
                        likelihood_t[t] += likelihood[t].item() / num_data

                    elif t < self.layer_num:
                        (10.0 * loss_list[t]).backward(retain_graph=True)
                        loss_t[t] += loss_list[t].item() / num_data
                        likelihood_t[t] += likelihood[t].item() / num_data
                    else:
                        (0 * loss_list[t]).backward(retain_graph=True)
                        loss_t[t] += loss_list[t].item() / num_data
                        likelihood_t[t] += likelihood[t].item() / num_data

                for para in self.model.parameters():
                    flag = torch.sum(torch.isnan(para))

                if (flag == 0):
                    nn.utils.clip_grad_norm_(self.model.decoder.parameters(), max_norm=100, norm_type=2)
                    self.decoder_optimizer.step()
                    self.decoder_optimizer.zero_grad()

            if epoch % 1 == 0:
                for t in range(self.layer_num + 1):
                    print('epoch {}|{}, layer {}|{}, loss: {}, likelihood: {}, lb: {}'.format(epoch, self.epochs, t,
                                                                                              self.layer_num,
                                                                                              loss_t[t]/2,
                                                                                              (likelihood_t[t]*0.5),
                                                                                              loss_t[t]/2))
                

            self.model.eval()

            if epoch % 10 == 0:
                test_likelihood, test_ppl = self.test(train_data_loader)
                save_checkpoint({'state_dict': self.model.state_dict(), 'epoch': epoch}, self.save_path, True)
                print('epoch {}|{}, test_ikelihood,{}, ppl: {}'.format(epoch, self.epochs, test_likelihood, test_ppl))


    def test(self, data_loader):
        self.model.eval()

        likelihood_t = 0
        num_data = len(data_loader)
        ppl_total = 0

        for i, (train_data, test_data) in enumerate(data_loader):
            train_data = torch.tensor(train_data, dtype = torch.float).to(self.args.device)
            test_data = torch.tensor(test_data, dtype=torch.float).to(self.args.device)
            # test_label = torch.tensor(test_label, dtype=torch.long).cuda()

            with torch.no_grad():
                ppl = self.model.test_ppl(train_data, test_data)
                # likelihood_total += ret_dict["likelihood"][0].item() / num_data
                ppl_total += ppl.item() / num_data

            # re_x, theta, loss_list, likelihood = self.model(test_data)
            # likelihood_t += likelihood[0].item() / num_data

        # save_checkpoint({'state_dict': self.model.state_dict(), 'epoch': epoch}, self.save_path, True)

        return likelihood_t, ppl_total

    def train_for_clustering(self, train_data_loader):
        self.model.to(self.args.device)
        KL_batch = [0] * (self.layer_num)
        likelihood_batch = [0] * (self.layer_num)
        division_likeli_loss_batch = 0.0
        num_data = len(train_data_loader)

        for train_data, train_label in tqdm.tqdm(train_data_loader):
            train_data = train_data.to(self.args.device)
            # self.optimizer.zero_grad()
            # self.decoder_optimizer.zero_grad()
            # self.layer_alpha_optimizer.zero_grad()

            # update inference network
            self.model.h_encoder.train()
            self.model.shape_encoder.train()
            self.model.scale_encoder.train()
            self.model.decoder.eval()

            ret_dict = self.model(train_data)
            KL_loss = ret_dict[2][1:]
            Likelihood = ret_dict[3][1:]

            Q_value = torch.tensor(0., device=self.args.device)
            for t in range(self.layer_num):  # from layer layer_num-1-step to 0
                Q_value += 10. * (Likelihood[t] + KL_loss[t])
            Q_value.backward()

            # Q_value = 10 * (ret_dict['division_likeli_loss'] + Likelihood[0])
            # for t in range(self.layer_num - 1):
            #     Q_value += 10 * self.args.kl_weight*KL_loss[t]
            # Q_value.backward()

            for para in self.model.parameters():
                flag = torch.sum(torch.isnan(para))

            if (flag == 0):
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=100, norm_type=2)
                self.optimizer.step()
                #self.layer_alpha_optimizer.step()
                #self.layer_alpha_optimizer.zero_grad()
                self.optimizer.zero_grad()

            # update generative network
            self.model.h_encoder.eval()
            self.model.shape_encoder.eval()
            self.model.scale_encoder.eval()
            self.model.decoder.train()

            ret_dict = self.model(train_data)
            KL_loss = ret_dict[2][1:]
            Likelihood = ret_dict[3][1:]

            Q_value = torch.tensor(0., device=self.args.device)
            for t in range(self.layer_num):  # from layer layer_num-1-step to 0
                Q_value += 10. * (Likelihood[t] + KL_loss[t])
            Q_value.backward()

            # Q_value = 10 * (ret_dict['division_likeli_loss'] + Likelihood[0])
            # for t in range(self.layer_num - 1):
            #     Q_value += 10 * self.args.kl_weight*KL_loss[t]
            # Q_value.backward()

            for para in self.model.parameters():
                flag = torch.sum(torch.isnan(para))

            if (flag == 0):
                nn.utils.clip_grad_norm_(self.model.decoder.parameters(), max_norm=100, norm_type=2)
                self.decoder_optimizer.step()
                #self.layer_alpha_optimizer.step()
                #self.layer_alpha_optimizer.zero_grad()
                self.decoder_optimizer.zero_grad()

            # update alpha
            self.model.train()  # require_grad for alpha
            # self.model.h_encoder.eval()
            # self.model.shape_encoder.eval()
            # self.model.scale_encoder.eval()
            # self.model.decoder.eval()

            ret_dict = self.model(train_data)

            #division_loss = ret_dict["division_likeli_loss"]
            #division_loss.backward()

            for para in self.model.parameters():
                flag = torch.sum(torch.isnan(para))

            if (flag == 0):
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=100, norm_type=2)
                self.optimizer.step()
                #self.layer_alpha_optimizer.step()
                self.decoder_optimizer.step()
                self.decoder_optimizer.zero_grad()
                #self.layer_alpha_optimizer.zero_grad()
                self.optimizer.zero_grad()

    def load_model(self):
        checkpoint = torch.load(self.save_path)
        self.GBN_models.load_state_dict(checkpoint['state_dict'])

    def extract_theta(self, data_loader):
        self.model.eval()
        test_theta_batch = []
        test_label_batch = []
        for test_data, test_label in tqdm.tqdm(data_loader):
            test_data = test_data.type(torch.float).to(self.args.device)
            test_label = test_label.to(self.args.device)
            ret_dict = self.model.forward(test_data)
            test_theta_batch.append([ret_dict[1][i].detach().cpu().numpy() for i in range(self.layer_num)])
            test_label_batch.append(test_label.detach().cpu().numpy())

        return test_theta_batch, test_label_batch

    def vis(self):
        # layer1
        w_1 = torch.mm(self.GBN_models[0].decoder.rho, torch.transpose(self.GBN_models[0].decoder.alphas, 0, 1))
        phi_1 = torch.softmax(w_1, dim=0).cpu().detach().numpy()

        index1 = range(100)
        dic1 = phi_1[:, index1[0:49]]
        # dic1 = phi_1[:, :]
        fig7 = plt.figure(figsize=(10, 10))
        for i in range(dic1.shape[1]):
            tmp = dic1[:, i].reshape(28, 28)
            ax = fig7.add_subplot(7, 7, i + 1)
            ax.axis('off')
            ax.set_title(str(index1[i] + 1))
            ax.imshow(tmp)

        # layer2
        w_2 = torch.mm(self.GBN_models[1].decoder.rho, torch.transpose(self.GBN_models[1].decoder.alphas, 0, 1))
        phi_2 = torch.softmax(w_2, dim=0).cpu().detach().numpy()
        index2 = range(49)
        dic2 = np.matmul(phi_1, phi_2[:, index2[0:49]])
        #dic2 = np.matmul(phi_1, phi_2[:, :])
        fig8 = plt.figure(figsize=(10, 10))
        for i in range(dic2.shape[1]):
            tmp = dic2[:, i].reshape(28, 28)
            ax = fig8.add_subplot(7, 7, i + 1)
            ax.axis('off')
            ax.set_title(str(index2[i] + 1))
            ax.imshow(tmp)

        # layer2
        w_3 = torch.mm(self.GBN_models[2].decoder.rho, torch.transpose(self.GBN_models[2].decoder.alphas, 0, 1))
        phi_3 = torch.softmax(w_3, dim=0).cpu().detach().numpy()
        index3 = range(32)

        dic3 = np.matmul(np.matmul(phi_1, phi_2), phi_3[:, index3[0:32]])
        #dic3 = np.matmul(np.matmul(phi_1, phi_2), phi_3[:, :])

        fig9 = plt.figure(figsize=(10, 10))
        for i in range(dic3.shape[1]):
            tmp = dic3[:, i].reshape(28, 28)
            ax = fig9.add_subplot(7, 7, i + 1)
            ax.axis('off')
            ax.set_title(str(index3[i] + 1))
            ax.imshow(tmp)

        plt.show()

    def get_voc(self, voc_path):
        if type(voc_path) == 'str':
            voc = []
            with open(voc_path) as f:
                lines = f.readlines()
            for line in lines:
                voc.append(line.strip())
            return voc
        else:
            return voc_path

    def vision_phi(self, Phi, outpath='phi_output', top_n=10):
        if self.voc is not None:
            #print(self.args.dataset_dir)
            dataset_dir = self.args.dataset_dir
            if dataset_dir[-8:] == '20ng.pkl':
                dataset_path = './freqdata/20ng/'
            if dataset_dir[-6:] == 'R8.pkl':
                dataset_path = './freqdata/R8/'
            if dataset_dir[-8:] == 'zhdd.pkl':
                dataset_path = './freqdata/zhdd/'
            if dataset_dir[-9:] == 'trump.pkl':
                dataset_path = './freqdata/trump/'
            if dataset_dir[-15:] == 'threetopics.pkl':
                dataset_path = './freqdata/threetopics/'
            if dataset_dir[-10:] == 'sample.pkl':
                dataset_path = './freqdata/dbpedia_sample/'
            if dataset_dir[-14:] == 'googlenews.pkl':
                dataset_path = './freqdata/googlenews/'
            if dataset_dir[-12:] == 'Snippets.pkl':
                dataset_path = './freqdata/SearchSnippets/'
            if dataset_dir[-13:] == 'TagMyNews.pkl':
                dataset_path = './freqdata/TagMyNews/'
            if dataset_dir[-9:] == 'Tweet.pkl':
                dataset_path = './freqdata/Tweet/'
            if dataset_dir[-10:] == 'agnews.pkl':
                dataset_path = './freqdata/agnews/'

            with open(dataset_path + 'co_occurrence_matrix.pkl', 'rb') as f:
                co_occurrence_matrix = pickle.load(f)

            with open(dataset_path + 'row_sums.pkl', 'rb') as f:
                row_sums = pickle.load(f)

            with open(dataset_path + 'word_frequencies.pkl', 'rb') as f:
                word_times = pickle.load(f)
            if not os.path.exists(outpath):
                os.makedirs(outpath)
            phi = 1
            all_topic_words = []
            #for num, phi_layer in enumerate(Phi):
            for num, phi_layer in enumerate(Phi[:1]):
                phi = np.dot(phi, phi_layer)
                phi_k = phi.shape[1]
                path = os.path.join(outpath, 'phi' + str(num) + '.txt')
                f = open(path, 'w')
                for each in range(phi_k):
                    top_n_words, top_n_words2, top_n_freqs = self.get_top_n(phi[:, each], top_n)
                    f.write(top_n_words)
                    f.write('\n')
                    all_topic_words.append(top_n_words2)
                f.close()
                all_topic_words_flat = [word for topic_words in all_topic_words for word in topic_words]
                #print(len(all_topic_words_flat))
                word_count = {}
                for word in all_topic_words_flat:
                    word_count[word] = word_count.get(word, 0) + 1
                unique_words = [word for word, count in word_count.items() if count == 1]

                freq_path = os.path.join(outpath, 'freqs' + str(num) + '.txt')
                freqs_file = open(freq_path,'w')
                # 计算每个主题的TC和TD值
                topic_coherence = []
                topic_diversity = []
                topic_quality = []
                if self.use_pal == False:
                    for topic_id in range(phi_k):
                        #print(topic_id)
                        freqs_file.write(f"Topic: {topic_id}\n")
                        _, top_n_words, top_n_freqs = self.get_top_n(phi[:, topic_id], top_n)
                        #print(top_n_words)
                        top_n_indices = []
                        for word in top_n_words:
                            if word in self.voc:
                                top_n_indices.append(self.voc.index(word))
                            else:
                                top_n_indices.append(None)

                        unique_words_count = 0
                        for word in top_n_words:
                            if word in unique_words:
                                unique_words_count += 1

                        tc = 0
                        for i in range(len(top_n_words)):
                            freqs_file.write(f"Word {i}\n")
                            for j in range(i + 1, len(top_n_words)):
                                if top_n_indices[i] < top_n_indices[j]:
                                    co_occurrence = co_occurrence_matrix.toarray()[top_n_indices[i]][top_n_indices[j]]
                                    #co_occurrence_freq = co_occurrence / row_sums[top_n_indices[i]]
                                else :
                                    co_occurrence = co_occurrence_matrix.toarray()[top_n_indices[j]][top_n_indices[i]]
                                    #co_occurrence_freq = co_occurrence / row_sums[top_n_indices[j]]
                                co_occurrence_freq = co_occurrence / self.count
                                co_freq = np.squeeze(co_occurrence_freq)
                                co_freq = float(co_freq)
                                freqs_file.write(f"co_freq of word '{top_n_words[i]}' and '{top_n_words[j]}' is: {co_freq}\n")
                                #print("co_freq of two words: ", co_freq)
                                ind_i = self.find_index(self.voc, top_n_words[i])
                                ind_j = self.find_index(self.voc, top_n_words[j])
                                freq_i = word_times[ind_i] / self.count
                                freq_i = np.squeeze(freq_i)
                                freq_i = float(freq_i)
                                freq_j = word_times[ind_j] / self.count
                                freq_j = np.squeeze(freq_j)
                                freq_j = float(freq_j)
                                if co_freq == 0:
                                    npmi = 0
                                else :
                                    pmi = np.log( co_freq / (freq_i * freq_j) )
                                    npmi = - pmi / np.log(co_freq)

                                tc += npmi
                            #print("freq from dataset: ", freq_i)
                            freqs_file.write(f"freq from dataset: {freq_i}\n")
                            #print("freq from model phi: ", top_n_freqs[i])
                            freqs_file.write(f"freq from model phi: {top_n_freqs[i]}\n")
                        tc /= (top_n * (top_n - 1) / 2)
                        td = unique_words_count / top_n
                        tq = tc * td

                        topic_coherence.append(tc)
                        topic_diversity.append(td)
                        topic_quality.append(tq)

                        print(f"Topic {topic_id + 1}: TC = {tc}, TD = {td}, TQ = {tq} ")
                        
                else:
                    for topic_id in range(phi_k):
                        #print(topic_id)
                        freqs_file.write(f"Topic: {topic_id}\n")
                        top_words, top_n_words, top_n_freqs = self.get_top_n(phi[:, topic_id], top_n)
                        #print(top_n_words)
                        top_n_indices = []
                        for word in top_n_words:
                            if word in self.voc:
                                top_n_indices.append(self.voc.index(word))
                            else:
                                top_n_indices.append(None)

                        unique_words_count = 0
                        for word in top_n_words:
                            if word in unique_words:
                                unique_words_count += 1
                        
                        tc = palmetto.get_coherence(top_n_words, 'npmi')
                        td = unique_words_count / top_n
                        tq = tc * td
                        topic_coherence.append(tc)
                        topic_diversity.append(td)
                        topic_quality.append(tq)
                        
                        print(f"Topic {topic_id + 1}: TC = {tc}, TD = {td}, TQ = {tq} ")

                freqs_file.close()
                # 将TC和TD值写入文件
                tc_path = os.path.join(outpath, f"topic_coherence_{num}.txt")
                td_path = os.path.join(outpath, f"topic_diversity_{num}.txt")
                tq_path = os.path.join(outpath, f"topic_quality_{num}.txt")
                np.savetxt(tc_path, topic_coherence, fmt="%.4f")
                np.savetxt(td_path, topic_diversity, fmt="%.4f")
                np.savetxt(tq_path, topic_quality, fmt="%.4f")
                tc_avg = np.mean(topic_coherence)
                td_avg = np.mean(topic_diversity)
                tq_avg = tc_avg * td_avg
                print(f"Result of this epoch: TC: {tc_avg}, TD: {td_avg}, TQ: {tq_avg}")
                
        else:
            print('voc need !!')

    def get_top_n(self, phi, top_n):
        top_n_words = ''
        top_n_words2 = []
        top_n_freqs = []
        idx = np.argsort(-phi)
        for i in range(top_n):
            index = idx[i]
            top_n_words += self.voc[index]
            top_n_words += ' '
            top_n_words2.append(self.voc[index])
            top_n_freqs.append(phi[index])
        return top_n_words, top_n_words2, top_n_freqs

    #def vis_txt(self, outpath='phi_output'):
    def vis_txt(self, epoch):
        phi = []
        for t in range(self.layer_num):
            #phi.append(self.model.decoder[t].get_phi().cpu().detach().numpy())
            w_t = torch.mm(self.model.decoder[t].rho, torch.transpose(self.model.decoder[t].alphas, 0, 1))
            phi_t = torch.softmax(w_t, dim=0).cpu().detach().numpy()
            phi.append(phi_t)

        if epoch % self.save_phi_every_epoch == 0:
            print("Epoch: {}".format(epoch))
        #self.vision_phi(phi, outpath=outpath)
            self.vision_phi(phi, os.path.join(self.save_path, "phi", f"{epoch:03d}"))

    def find_index(self, arr, target):
        for i in range(len(arr)):
            if arr[i] == target:
                return i
        return -1
