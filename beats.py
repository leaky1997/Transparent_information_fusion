import paramiko
import time
import threading

def send_heartbeat(ssh):
    while True:
        ssh.exec_command("\n")  # 发送一个空命令作为心跳
        time.sleep(60)  # 每隔5秒发送一次心跳

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('166.111.52.234', username='user', password='thu@a413')

# 在一个新的线程中发送心跳，以便主线程可以继续执行其他任务
threading.Thread(target=send_heartbeat, args=(ssh,)).start()