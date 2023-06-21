import torch.optim as optim
from torch.distributions.multivariate_normal import MultivariateNormal
import ot
from utils import *
from nets import DQN
import time as Clock

start = Clock.time()

####### One-dimensional case #########
# with parameter constraint
Trunc_flag = True
# No. of gradient descent steps (G)
N_OPT = 20
# No. of sample paths (N)
smp_size = 2000
# Sample size for empirical OT (B)
in_sample_size = 50

time_horizon = 8
x_dim = 1
y_dim = 1
x_vol = 1.0
y_vol = 0.5
x_init = 1.0
y_init = 2.0


###### Multidimensional case #########
## no parameter constraint
# Trunc_flag = False
# time_horizon = 5
# x_dim = 5
# y_dim = 5
# x_vol = 1.1
# y_vol = 0.1
# x_init = 1.0
# y_init = 2.0
# N_OPT = 400
# smp_size = 4000
# in_sample_size = 300


final_result = np.zeros(N_INSTANCE)

for n_ins in range(N_INSTANCE):

    val_hist = np.zeros(time_horizon+1)
    loss_hist = np.zeros(time_horizon+1)

    memory = Memory(MEM_SIZE)
    policy_net = DQN(x_dim, y_dim, time_horizon).to(device)
    target_net = DQN(x_dim, y_dim, time_horizon).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    # optimizer = optim.SGD(policy_net.parameters(), lr=0.1, momentum=0.9)
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-2) # weight_decay=1e-3)

    x_path_pool = torch.zeros(smp_size, time_horizon+1, x_dim, device=device)
    y_path_pool = torch.zeros(smp_size, time_horizon+1, y_dim, device=device)
    x_path_pool[:, 0, :] = x_init
    y_path_pool[:, 0, :] = y_init

    for smp_id in range(smp_size):
        # sample many paths in advance
        for t in range(1, time_horizon + 1):
            x_path_pool[smp_id, t, :] = x_path_pool[smp_id, t - 1, :] + x_vol * torch.randn(x_dim, device=device)
            y_path_pool[smp_id, t, :] = y_path_pool[smp_id, t - 1, :] + y_vol * torch.randn(y_dim, device=device)

    for time in range(time_horizon, -1, -1):

        for smp_id in range(smp_size):
            x_mvn = MultivariateNormal(loc=x_path_pool[smp_id, time, :], covariance_matrix=torch.eye(x_dim, device=device)*x_vol**2)
            y_mvn = MultivariateNormal(loc=y_path_pool[smp_id, time, :], covariance_matrix=torch.eye(y_dim, device=device)*y_vol**2)
            next_x = x_mvn.sample((in_sample_size,))
            next_y = y_mvn.sample((in_sample_size,))

            x_batch = torch.repeat_interleave(next_x, repeats=in_sample_size, dim=0)
            y_batch = torch.tile(next_y, (in_sample_size, 1))
            l2_mat = torch.sum((x_batch - y_batch)**2, dim=1)

            if time == time_horizon:
                expected_v = 0.0
            elif time == time_horizon-1:
                min_obj = l2_mat.reshape(in_sample_size, in_sample_size)
                expected_v = ot.emd2(np.ones(in_sample_size) / in_sample_size, np.ones(in_sample_size) / in_sample_size,
                                     min_obj.detach().cpu().numpy())
            else:
                val = target_net(torch.ones(x_batch.shape[0], 1, device=device)*(time+1.0), x_batch, y_batch).reshape(-1)
                min_obj = (l2_mat + DISCOUNT*val).reshape(in_sample_size, in_sample_size)
                expected_v = ot.emd2(np.ones(in_sample_size)/in_sample_size, np.ones(in_sample_size)/in_sample_size,
                                     min_obj.detach().cpu().numpy())

            memory.push(torch.tensor([time], dtype=torch.float32, device=device), x_path_pool[smp_id, time, :],
                        y_path_pool[smp_id, time, :], torch.tensor([expected_v], device=device))

        # Optimize at time t
        for opt_step in range(N_OPT):
            loss = optimize_model(policy_net, memory, optimizer, Trunc_flag)
            if Trunc_flag:
                with torch.no_grad():
                    for param in policy_net.parameters():
                        ## param.add_(torch.randn(param.size(), device=device)/50)
                        param.clamp_(-1.0, 1.0)
            if loss:
                loss_hist[time] += loss.detach().cpu().item()


        loss_hist[time] /= N_OPT

        # update target network
        target_net.load_state_dict(policy_net.state_dict())
        # test initial value
        val = target_net(torch.ones(1, 1, device=device)*0.0, x_path_pool[0, 0, :].reshape(1, x_dim),
                         y_path_pool[0, 0, :].reshape(1, y_dim)).reshape(-1)
        val_hist[time] = val

        # empty memory
        memory.clear()
        print('Time step', time, 'Loss', loss_hist[time])

        # print('Shift vector in the last layer:', target_net.linear3.bias.sum().item())


    # for name, param in target_net.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.data)


    print('Instance', n_ins)
    # print('Time elapsed', end - start)
    print('Last values', val_hist[0])
    final_result[n_ins] = val_hist[0]

print('All final value:', final_result)
print('Final mean:', final_result.mean())
print('Final std:', final_result.std())
end = Clock.time()
print('Average time for one instance:', (end-start)/N_INSTANCE)
# plt.figure(figsize=(8, 6))
# plt.plot(val_hist)
# plt.xlabel('Steps', fontsize=16)
# plt.ylabel(r'$V_0$', fontsize=16)
# # plt.tick_params(axis = 'both', which = 'major', labelsize = 16)
# plt.legend(bbox_to_anchor=(1, 1), title='', fontsize=16, title_fontsize=16)
# plt.savefig('conti_val.pdf', format='pdf', dpi=1000, bbox_inches='tight', pad_inches=0.1)
# plt.show()


# plt.figure(figsize=(8, 6))
# plt.plot(loss_hist)
# plt.xlabel('Steps', fontsize=16)
# plt.ylabel('Loss', fontsize=16)
# plt.savefig('conti_loss.pdf', format='pdf', dpi=1000, bbox_inches='tight', pad_inches=0.1)
# plt.show()
