import serial
import datetime
import math
import numpy as np


class Point:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return 'x:%.2f y:%.2f z:%.2f' % (self.x, self.y, self.z)


(Angle, Acc, Gyro) = (Point,) * 3

filter_cnt = 0
FILTER_LENGTH = 20
acc_buff = [Acc] * FILTER_LENGTH
filled = False

(acc_offset_x, acc_offset_y, acc_offset_z) = (0,) * 3
(gyro_offset_x, gyro_offset_y, gyro_offset_z) = (0,) * 3


def bytes2int(x):
    return int.from_bytes(x, byteorder='big', signed=True)


def get_angle(x, y):
    lx = np.sqrt(x.dot(x))
    ly = np.sqrt(y.dot(y))
    cos_angle = x.dot(y) / (lx * ly)
    res = np.arccos(cos_angle)
    return res * 180 / np.pi


def get_gyro(data):
    # 温度buff
    temp = bytes2int(data[7] + data[6])
    # 三轴角速度buff
    x = bytes2int(data[1] + data[0]) - gyro_offset_x
    y = bytes2int(data[3] + data[2]) - gyro_offset_y
    z = bytes2int(data[5] + data[4]) - gyro_offset_z
    # print(x, y, z)
    return x, y, z, temp


def angular_velocity(data):
    x, y, z, temp = get_gyro(data)
    # print('x: %d y: %d z: %d' % (x, y, z))
    # 设置三轴角速度阈值，防止静止状态下漂移
    # threshold = 5
    # if -threshold <= x <= threshold:
    #     x = 0
    # if -threshold <= y <= threshold:
    #     y = 0
    # if -threshold <= z <= threshold:
    #     z = 0
    temp = 35 + (1.0 * temp + 13200) / 280
    x = x / 14.375 * math.pi / 180
    y = y / 14.375 * math.pi / 180
    z = z / 14.375 * math.pi / 180
    return x, y, z, temp


def get_acc(data):
    ax = bytes2int(data[-5] + data[-6]) - acc_offset_x
    ay = bytes2int(data[-3] + data[-4]) - acc_offset_y
    az = bytes2int(data[-1] + data[-2]) - acc_offset_z
    return ax, ay, az


def acc_filter(tup):
    ax, ay, az = tup
    global filter_cnt, filled, acc_buff
    acc_buff[filter_cnt] = Acc(ax, ay, az)
    filter_cnt += 1
    if filter_cnt >= FILTER_LENGTH:
        filter_cnt = 0
        filled = True

    length = FILTER_LENGTH if filled else filter_cnt

    tsum = Acc()
    for i in range(length):
        tsum.x += acc_buff[i].x
        tsum.y += acc_buff[i].y
        tsum.z += acc_buff[i].z

    return tsum.x / length, tsum.y / length, tsum.z / length


def acc(data):
    ax, ay, az = acc_filter(data)
    # pitch = math.atan2(ay, math.sqrt(ax * ax + az * az)) * 180 / math.pi
    # roll = math.atan2(-ax, az) * 180 / math.pi
    # print(pitch, roll, end='\r')
    ax = 0.039 * ax
    ay = 0.039 * ay
    az = 0.039 * az
    # print("ax: %.2f ay: %.2f az: %.2f" % (ax, ay, az))
    return ax, ay, az


q0, q1, q2, q3 = (1.0, 0.0, 0.0, 0.0)
his_q0, his_q1, his_q2, his_q3 = (1.0, 0.0, 0.0, 0.0)
angle = Angle()
v = np.array([1.0, 0.0, 0.0])
# x、y、z轴的比例误差
ex_int, ey_int, ez_int = (0.0, 0.0, 0.0)
IMU_KP = 1.5
IMU_KI = 0.0005


def imu_update(c, gx, gy, gz, ax, ay, az, T):
    global q0, q1, q2, q3, his_q0, his_q1, his_q2, his_q3, ex_int, ey_int, ez_int, IMU_KP, IMU_KI
    half_T = T / 2

    if ax * ay * az == 0:
        print('ax * ay * az is 0')
        return

    norm = math.sqrt(ax * ax + ay * ay + az * az)
    ax /= norm
    ay /= norm
    az /= norm

    vx = (his_q1 * his_q3 - his_q0 * his_q2) * 2
    vy = (his_q0 * his_q1 + his_q2 * his_q3) * 2
    vz = his_q0 * his_q0 - his_q1 * his_q1 - his_q2 * his_q2 + his_q3 * his_q3

    # 向量外积再相减得到差分就是误差，两个单位向量的差积即为误差向量
    ex = (ay * vz - az * vy)
    ey = (az * vx - ax * vz)
    ez = (ax * vy - ay * vx)

    # 对误差进行PI计算
    ex_int = ex_int + ex * IMU_KI
    ey_int = ey_int + ey * IMU_KI
    ez_int = ez_int + ez * IMU_KI

    gx = gx + IMU_KP * ex + ex_int
    gy = gy + IMU_KP * ey + ey_int
    gz = gz + IMU_KP * ez + ez_int

    # 四元素的微分方程
    q0 = his_q0 + (-his_q1 * gx - his_q2 * gy - his_q3 * gz) * half_T
    q1 = his_q1 + (his_q0 * gx + his_q2 * gz - his_q3 * gy) * half_T
    q2 = his_q2 + (his_q0 * gy - his_q1 * gz + his_q3 * gx) * half_T
    q3 = his_q3 + (his_q0 * gz + his_q1 * gy - his_q2 * gx) * half_T

    q0q0 = q0 * q0
    q0q1 = q0 * q1
    q0q2 = q0 * q2
    q0q3 = q0 * q3
    q1q1 = q1 * q1
    q1q2 = q1 * q2
    q1q3 = q1 * q3
    q2q2 = q2 * q2
    q2q3 = q2 * q3
    q3q3 = q3 * q3

    # 规范化Pitch、Roll轴四元数
    norm = math.sqrt(q0q0 + q1q1 + q2q2 + q3q3)
    q0 = q0 / norm
    q1 = q1 / norm
    q2 = q2 / norm
    q3 = q3 / norm

    # 放在b系中算角度
    # tv = np.array([2 * (q1q2 + q0q3), 1 - 2 * (q1q1 + q3q3), 2 * (q2q3 - q0q1)])

    # 放在n系中算角度，用于多陀螺仪情况
    # 起始向量(1, 0, 0)
    tv2 = np.array([1 - 2 * (q2q2 + q3q3), 2 * (q1q2 + q0q3), 2 * (q1q3 - q0q2)])
    # 起始向量(0, 0, 1)
    # tv2 = np.array([2 * (q1q3 + q0q2), 2 * (q2q3 - q0q1), 1 - 2 * (q1q1 + q2q2)])
    # 起始向量(0, 1, 0)
    # tv2 = np.array([2 * (q1q2 - q0q3), 1 - 2 * (q1q1 + q3q3), 2 * (q2q3 + q0q1)])

    h_tv2 = tv2.copy()
    h_tv2[2] = 0.0
    v_tv2 = tv2.copy()
    v_tv2[1] = 0.0

    # 求解欧拉角
    angle.x = math.atan2(2 * q2q3 + 2 * q0q1, -2 * q1q1 - 2 * q2q2 + 1)
    angle.y = math.asin(-2 * q1q3 + 2 * q0q2)
    angle.z = math.atan2(2 * q1q2 + 2 * q0q3, -2 * q2q2 - 2 * q3q3 + 1)

    print('A:%.2f H:%.2f V:%.2f R_x:%.2f' %
          (get_angle(v, tv2), get_angle(v, h_tv2), get_angle(v, v_tv2), angle.x * 57.3), end='\r')

    # 存储更替相应的四元数
    his_q0 = q0
    his_q1 = q1
    his_q2 = q2
    his_q3 = q3


ser = serial.Serial('COM5', 9600)
(wa, c, frame_len, data) = (None,) * 4
(pre_time, now_time, delta_time, start) = (None,) * 4


def init():
    global wa, c, frame_len, data, pre_time, now_time, delta_time, start
    (wa, c) = (0,) * 2
    frame_len = 20
    data = [0] * frame_len
    (pre_time, now_time, delta_time) = (None,) * 3
    start = False


def read_data():
    global wa, c, frame_len, data, pre_time, now_time, delta_time, start
    while True:
        if ser.read() == b'\xaa' and ser.read() == b'\xaa':
            if start:
                now_time = datetime.datetime.now()
                delta_time = (now_time - pre_time).microseconds / 1e6
                pre_time = now_time
            else:
                pre_time = datetime.datetime.now()
            cou = 0
            c = c + 1
            while True:
                t = ser.read()
                cou = cou + 1
                if t != b'\xee':
                    if 0 <= cou - 1 < frame_len:
                        data[cou - 1] = t
                else:
                    t = ser.read()
                    cou = cou + 1
                    if t == b'\xee':
                        cou = cou - 2
                        break
                    else:
                        if 0 <= cou - 2 < frame_len:
                            data[cou - 2] = b'\xee'
                        if 0 <= cou - 1 < frame_len:
                            data[cou - 1] = t
            if cou != frame_len:
                wa = wa + 1
                print(str(wa) + ' in ' + str(c))
            else:
                if start:
                    if filled:
                        return data
                    else:
                        acc_filter(get_acc(data))
            start = True


def data_offset(cnt):
    global ser, wa, c, frame_len, data, pre_time, now_time, delta_time, start
    global acc_offset_x, acc_offset_y, acc_offset_z, gyro_offset_x, gyro_offset_y, gyro_offset_z
    sum_acc = Acc(0.0, 0.0, 0.0)
    sum_gyro = Gyro(0.0, 0.0, 0.0)
    length = cnt

    while True:
        data_frame = read_data()
        t_Acc = Acc()
        t_Gyro = Gyro()
        (t_Acc.x, t_Acc.y, t_Acc.z) = get_acc(data)
        (t_Gyro.x, t_Gyro.y, t_Gyro.z, _) = get_gyro(data)
        sum_acc.x += t_Acc.x
        sum_acc.y += t_Acc.y
        # sum_acc.z += t_Acc.z
        sum_gyro.x += t_Gyro.x
        sum_gyro.y += t_Gyro.y
        sum_gyro.z += t_Gyro.z
        cnt -= 1
        if cnt % 10 == 0:
            print('offset left:', cnt)
        if cnt == 0:
            acc_offset_x = sum_acc.x / length
            acc_offset_y = sum_acc.y / length
            acc_offset_z = sum_acc.z / length
            gyro_offset_x = sum_gyro.x / length
            gyro_offset_y = sum_gyro.y / length
            gyro_offset_z = sum_gyro.z / length
            print('acc_offset:', acc_offset_x, acc_offset_y, acc_offset_z)
            print('gyro_offset:', gyro_offset_x, gyro_offset_y, gyro_offset_z)
            break


init()
data_offset(100)
init()


while True:
    data_frame = read_data()
    ax, ay, az = acc(get_acc(data_frame))
    gx, gy, gz, _ = angular_velocity(data_frame)
    imu_update(c, gx, gy, gz, ax, ay, az, delta_time)
