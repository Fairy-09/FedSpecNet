

import pickle
import socket
import time
import struct

def log(info):
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' ' + str(info))

Epoch = 10

def m1(*args):
    result = args[0].clone()  # 使用 clone() 而不是 deepcopy
    for i in range(1, len(args)):
        result += args[i]

    result /= len(args)
    for i in range(len(args)):
        args[i].copy_(result)  # 原地赋值，避免新建张量


def m2(*args):
    result = args[0].clone()
    for i in range(1, len(args)):
        result += args[i]

    result /= len(args)
    for i in range(len(args)):
        args[i].copy_(result)


def socket_udp_server():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host = '127.0.0.1'
    port = 7002
    s.bind((host, port))
    s.listen(5)
    print('waiting for connecting')

    for cnt in range(1, Epoch + 1):
        log(f"第{cnt}轮开始接收并计时")
        connected_socks = []
        res = []

        while len(connected_socks) < 5:  ##number 测试的时候修改这个数量
            try:
                s.settimeout(100)  # 设置较长的超时时间等待连接
                sock, addr = s.accept()
                log(f'Connection accepted from {addr}')

                data_length_bytes = sock.recv(4)
                if not data_length_bytes:
                    raise ValueError("未接收到数据长度信息")
                data_length = int.from_bytes(data_length_bytes, byteorder='big')

                received_data = b''
                while len(received_data) < data_length:
                    packet = sock.recv(data_length - len(received_data))
                    if not packet:
                        raise ConnectionError("连接中断")
                    received_data += packet

                tmp = pickle.loads(received_data)
                log('Received data: ...')
                if tmp['num'] == cnt:
                    connected_socks.append(sock)
                    res.append(tmp['model'])

            except socket.timeout:
                log("接收数据超时。")
                break  # 跳出循环，处理已接收的数据
            except Exception as e:
                log(f"接收数据时发生异常: {e}")

        if res:
            # 数据处理逻辑
            log(f"第{cnt}轮接收完毕，接收来自{len(res)}个节点的参数")
            # 假设m1和m2函数的数据处理逻辑如前所述            
            log("开始融合处理操作......")

            for m, n in zip(res[0].values(), res[1].values()):
                log(f"融合处理参数维度: {m.size()} 和 {n.size()}")
                if len(m.size()) == 1:
                    m1(m, n)
                elif len(m.size()) == 2:
                    m2(m, n)

            data = {}
            data['num'] = cnt
            data['model'] = res[0]
            log('第%d轮融合完毕，下发......' % cnt)
            data = pickle.dumps(data)



            # 发送ACK确认给所有节点
            for sock in connected_socks:
                try:
                    # # 先发送'ACK'确认信号
                    # ack_message = 'ACK'.encode()
                    # sock.sendall(ack_message)
                    
                    # 发送融合后的数据长度
                    data_length = len(data)
                    packed_length = struct.pack('!I', data_length)
                    sock.sendall(packed_length)
                    
                    # 发送实际融合后的数据
                    sock.sendall(data)
                except Exception as e:
                    log(f"发送数据时出错: {e}")
                finally:
                    # 确保每次发送完毕后关闭连接
                    sock.close()
            log('Data sent and connections closed.')

        # 清理资源，准备下一轮
        connected_socks.clear()

    s.close()
    log('All epochs completed, server shutdown.')

def main():
    socket_udp_server()

if __name__ == '__main__':
    main()


# import pickle
# import socket
# import time
# import struct

# def log(info):
#     print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' ' + str(info))

# Epoch = 10

# def m1(*args):
#     import copy
#     result = copy.deepcopy(args[0])
#     for i in range(1, len(args)):
#         for j in range(len(result)):
#             result[j] += args[i][j]

#     for i in range(len(args)):
#         for j in range(len(args[0])):
#             args[i][j] = result[j] / len(args)

# def m2(*args):
#     import copy
#     result = copy.deepcopy(args[0])
#     for i in range(1, len(args)):
#         # print(args[i])
#         for j in range(len(result)):
#             for k in range(len(result[0])):
#                 result[j][k] += args[i][j][k]

#     for i in range(len(args)):
#         for j in range(len(result)):
#             for k in range(len(result[0])):
#                 args[i][j][k] = result[j][k] / len(args)


# def socket_udp_server():
#     s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     host = '127.0.0.1'
#     port = 7002
#     s.bind((host, port))
#     s.listen(5)
#     print('waiting for connecting')

#     for cnt in range(1, Epoch + 1):
#         log(f"第{cnt}轮开始接收并计时")
#         connected_socks = []
#         res = []

#         while len(connected_socks) < 2:  ##number 测试的时候修改这个数量
#             try:
#                 s.settimeout(100)  # 设置较长的超时时间等待连接
#                 sock, addr = s.accept()
#                 log(f'Connection accepted from {addr}')

#                 data_length_bytes = sock.recv(4)
#                 if not data_length_bytes:
#                     raise ValueError("未接收到数据长度信息")
#                 data_length = int.from_bytes(data_length_bytes, byteorder='big')

#                 received_data = b''
#                 while len(received_data) < data_length:
#                     packet = sock.recv(data_length - len(received_data))
#                     if not packet:
#                         raise ConnectionError("连接中断")
#                     received_data += packet

#                 tmp = pickle.loads(received_data)
#                 log('Received data: ...')
#                 if tmp['num'] == cnt:
#                     connected_socks.append(sock)
#                     res.append(tmp['model'])

#             except socket.timeout:
#                 log("接收数据超时。")
#                 break  # 跳出循环，处理已接收的数据
#             except Exception as e:
#                 log(f"接收数据时发生异常: {e}")

#         if res:
#             # 数据处理逻辑
#             log(f"第{cnt}轮接收完毕，接收来自{len(res)}个节点的参数")
#             # 假设m1和m2函数的数据处理逻辑如前所述
            
#             log("开始融合处理操作......")

#             for m, n in zip(res[0].values(), res[1].values()):
#                 if len(m.size()) == 1:
#                     m1(m, n)
#                 elif len(m.size()) == 2:
#                     m2(m, n)

#             data = {}
#             data['num'] = cnt
#             data['model'] = res[0]
#             log('第%d轮融合完毕，下发......' % cnt)
#             data = pickle.dumps(data)



#             # 发送ACK确认给所有节点
#             for sock in connected_socks:
#                 try:
#                     # # 先发送'ACK'确认信号
#                     # ack_message = 'ACK'.encode()
#                     # sock.sendall(ack_message)
                    
#                     # 发送融合后的数据长度
#                     data_length = len(data)
#                     packed_length = struct.pack('!I', data_length)
#                     sock.sendall(packed_length)
                    
#                     # 发送实际融合后的数据
#                     sock.sendall(data)
#                 except Exception as e:
#                     log(f"发送数据时出错: {e}")
#                 finally:
#                     # 确保每次发送完毕后关闭连接
#                     sock.close()
#             log('Data sent and connections closed.')

#         # 清理资源，准备下一轮
#         connected_socks.clear()

#     s.close()
#     log('All epochs completed, server shutdown.')

# def main():
#     socket_udp_server()

# if __name__ == '__main__':
#     main()
