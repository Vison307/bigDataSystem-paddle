import psutil
import time
from matplotlib import pyplot as plt
import numpy as np


def get_key():
    key_info = psutil.net_io_counters(pernic=True).keys()  # 获取网卡名称

    recv = {}
    sent = {}

    for key in key_info:
        recv.setdefault(key, psutil.net_io_counters(pernic=True).get(key).bytes_recv)  # 各网卡接收的字节数
        sent.setdefault(key, psutil.net_io_counters(pernic=True).get(key).bytes_sent)  # 各网卡发送的字节数

    return key_info, recv, sent


def get_rate(func):
    key_info, old_recv, old_sent = func()  # 上一秒收集的数据

    time.sleep(1)

    key_info, now_recv, now_sent = func()  # 当前所收集的数据

    net_in = {}
    net_out = {}

    for key in key_info:
        net_in.setdefault(key, (now_recv.get(key) - old_recv.get(key)) / (1024 * 1024))  # 每秒接收速率
        net_out.setdefault(key, (now_sent.get(key) - old_sent.get(key)) / (1024 * 1024))  # 每秒发送速率

    return key_info, net_in, net_out


if __name__ == '__main__':
    flow_in = []
    flow_out = []
    throughtput = []
    cnt = 0
    while True:
        cnt += 1
        try:
            key_info, net_in, net_out = get_rate(get_key)
            for key in key_info:
                if key == "meth1042":
                    print('%s\nInput:\t %-5sMB/s\nOutput:\t %-5sMB/s\n' % (key, net_in.get(key), net_out.get(key)))
                    flow_in.append(net_in.get(key))
                    flow_out.append(net_out.get(key))
                    throughtput.append(net_in.get(key) + net_out.get(key))

                    if cnt % 120 == 0:
                        plt.figure(dpi=300, figsize=(6, 4))
                        plt.plot(flow_in, 'm.-.', label='ax2', linewidth=1)
                        plt.title("Inbound Traffic")
                        plt.xlabel("seconds")
                        plt.ylabel("MByte/s")
                        plt.savefig("flow_in_{}.png".format(cnt))

                        plt.figure(dpi=300, figsize=(6, 4))
                        plt.plot(flow_out, 'm.-.', label='ax2', linewidth=1)
                        plt.title("Outbound Traffic")
                        plt.xlabel("seconds")
                        plt.ylabel("MByte/s")
                        plt.savefig("flow_out_{}.png".format(cnt))

                        plt.figure(dpi=300, figsize=(6, 4))
                        plt.plot(throughtput, 'm.-.', label='ax2', linewidth=1)
                        plt.title("Throughput")
                        plt.xlabel("seconds")
                        plt.ylabel("MByte/s")
                        plt.savefig("Throughput_{}.png".format(cnt))

        except KeyboardInterrupt:

            plt.figure(dpi=300, figsize=(6, 4))
            plt.plot(flow_in, 'm.-.', label='ax2', linewidth=1)
            plt.title("Inbound Traffic")
            plt.xlabel("seconds")
            plt.ylabel("MByte/s")
            plt.savefig("flow_in.png")

            plt.figure(dpi=300, figsize=(6, 4))
            plt.plot(flow_out, 'm.-.', label='ax2', linewidth=1)
            plt.title("Outbound Traffic")
            plt.xlabel("seconds")
            plt.ylabel("MByte/s")
            plt.savefig("flow_out.png")

            plt.figure(dpi=300, figsize=(6, 4))
            plt.plot(throughtput, 'm.-.', label='ax2', linewidth=1)
            plt.title("Throughput")
            plt.xlabel("seconds")
            plt.ylabel("MByte/s")
            plt.savefig("Throughput.png")
            
            with open('throughput.txt', 'w') as f:
                f.write(f'flow_in: {flow_in}, mean: {np.mean(flow_in): .4f}\n\n')
                f.write(f'flow_out: {flow_out}, mean: {np.mean(flow_out): .4f}\n\n')
                f.write(f'throughtput: {throughtput}, mean: {np.mean(throughtput): .4f}\n\n')