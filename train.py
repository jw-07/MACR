from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision import models
from data_loader import MTData



def train(args, model, mean_teacher, device, train_loader, test_loader, optimizer, epoch, dis):
    for batch_idx, ((data, data_fake, target_sim), (data_real, data_real_fake, target_real)) in enumerate(zip(train_loader, test_loader)):
        data, data_fake, target_sim, data_real, data_real_fake, target_real = data.to(device), data_fake.to(
            device), target_sim.to(device), data_real.to(device), data_real_fake.to(device), target_real.to(device)

        optimizer.zero_grad()

        output = model(data) # output of S
        output_fake = model(data_fake) # output of S2M
        output_real = model(data_real) # output of M



        # domain classifier

        x = torch.cat([data, data_real])
        x = x.to(device)

        domain_y = torch.cat([torch.ones(data.shape[0]),
                              torch.zeros(data_real.shape[0])])

        # # label  smoothing
        # lam = 0.5
        # domain_y = domain_y * (1 - lam) + lam / 2


        domain_y = domain_y.to(device)
        features = model(x).view(x.shape[0], -1)

        domain_preds = dis(features).squeeze()

        domain_loss = F.binary_cross_entropy_with_logits(domain_preds, domain_y)


        # forward pass with mean teacher
        # torch.no_grad() prevents gradients from being passed into mean teacher model
        with torch.no_grad():
            mean_t_output = mean_teacher(data_real_fake) # output of M2S
            # PLS
            mean_t_output_softmax = torch.softmax(mean_t_output, dim=1)
            mean_t_output_softmax_max, idx_label = torch.max(mean_t_output_softmax, dim=1)
            w = 0.3  # int(w*args.batch_size)  K=2
            if int(w * args.batch_size) <= idx_label.shape[0]:
                p_topk, idx_topk = torch.topk(mean_t_output_softmax_max, int(w * args.batch_size),
                                              dim=0)  # top-k label

            else:
                # minibatch < K
                idx_topk = torch.arange(0, idx_label.shape[0])  # 0 - minibatch_size




        # consistency loss

        loss0 = nn.CrossEntropyLoss()
        const_loss = loss0(output_real[idx_topk, :], idx_label[idx_topk])




        # set the consistency weight
        weight = 1
        loss1 = nn.CrossEntropyLoss()
        loss2 = nn.CrossEntropyLoss()

        loss_sim = loss1(output, target_sim)
        loss_sim_fake = loss2(output_fake, target_sim)

        loss = loss_sim + 0.5 * loss_sim_fake + 0.5 * domain_loss + weight * const_loss
        loss.backward()
        optimizer.step()




        # update mean teacher
        alpha = 0.95
        for mean_param, param in zip(mean_teacher.parameters(), model.parameters()):
            mean_param.data.mul_(alpha).add_(1 - alpha, param.data)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

            if (args.save_model):
                save_path = './checkpoint/' + str(epoch) + '_' + str(batch_idx * len(data)) + 'result.pth'
                torch.save(model.state_dict(), save_path)
                print('saved!')




