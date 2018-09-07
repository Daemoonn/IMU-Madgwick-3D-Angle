import serial
import datetime
import math
import numpy as np
from ctypes import *


class Point:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return 'x:%.2f y:%.2f z:%.2f' % (self.x, self.y, self.z)


(Angle, Acc, Gyro) = (Point,) * 3


class ITG3205:
    # IMU_KP = 1.5
    # IMU_KI = 0.0005

    def __init__(self, port_num):
        self.filter_cnt = 0
        self.FILTER_LENGTH = 1
        self.beta = 3
        self.acc_buff = [Acc] * self.FILTER_LENGTH
        self.filled = False
        (self.acc_offset_x, self.acc_offset_y, self.acc_offset_z) = (0,) * 3
        (self.gyro_offset_x, self.gyro_offset_y, self.gyro_offset_z) = (0,) * 3

        self.his_q0, self.his_q1, self.his_q2, self.his_q3 = (1.0, 0.0, 0.0, 0.0)
        self.angle = Angle()
        self.v = np.array([1.0, 0.0, 0.0])
        # x、y、z轴的比例误差
        self.ex_int, self.ey_int, self.ez_int = (0.0, 0.0, 0.0)

        self.ser = serial.Serial(port_num, 9600)

        (self.wa, self.c) = (0,) * 2
        self.frame_len = 20
        self.data = [0] * self.frame_len
        (self.pre_time, self.now_time, self.delta_time) = (None,) * 3
        self.start = False

        self.data_offset(100)
        self.re_init()

    def re_init(self):
        (self.wa, self.c) = (0,) * 2
        self.frame_len = 20
        self.data = [0] * self.frame_len
        (self.pre_time, self.now_time, self.delta_time) = (None,) * 3
        self.start = False

    @staticmethod
    def bytes2int(x):
        return int.from_bytes(x, byteorder='big', signed=True)

    @staticmethod
    def get_angle(x, y):
        lx = np.sqrt(x.dot(x))
        ly = np.sqrt(y.dot(y))
        cos_angle = x.dot(y) / (lx * ly)
        res = np.arccos(cos_angle)
        return res * 180 / np.pi

    @staticmethod
    def rad2angle(x):
        return x * 180 / np.pi

    def get_gyro(self, data):
        # 温度buff
        b2i = ITG3205.bytes2int
        temp = b2i(data[7] + data[6])
        # 三轴角速度buff
        x = b2i(data[1] + data[0]) - self.gyro_offset_x
        y = b2i(data[3] + data[2]) - self.gyro_offset_y
        z = b2i(data[5] + data[4]) - self.gyro_offset_z
        # print(x, y, z)
        return x, y, z, temp

    def angular_velocity(self, data):
        x, y, z, temp = self.get_gyro(data)
        temp = 35 + (1.0 * temp + 13200) / 280
        x = x / 14.375 * math.pi / 180
        y = y / 14.375 * math.pi / 180
        z = z / 14.375 * math.pi / 180
        return x, y, z, temp

    def get_acc(self, data):
        b2i = ITG3205.bytes2int
        ax = b2i(data[-5] + data[-6]) - self.acc_offset_x
        ay = b2i(data[-3] + data[-4]) - self.acc_offset_y
        az = b2i(data[-1] + data[-2]) - self.acc_offset_z
        return ax, ay, az

    def acc_filter(self, tup):
        ax, ay, az = tup
        # global filter_cnt, filled, acc_buff
        self.acc_buff[self.filter_cnt] = Acc(ax, ay, az)
        self.filter_cnt += 1
        if self.filter_cnt >= self.FILTER_LENGTH:
            self.filter_cnt = 0
            self.filled = True

        length = self.FILTER_LENGTH if self.filled else self.filter_cnt

        tsum = Acc()
        for i in range(length):
            tsum.x += self.acc_buff[i].x
            tsum.y += self.acc_buff[i].y
            tsum.z += self.acc_buff[i].z

        return tsum.x / length, tsum.y / length, tsum.z / length

    def read_data(self):
        # global wa, c, frame_len, data, pre_time, now_time, delta_time, start
        while True:
            if self.ser.read() == b'\xaa' and self.ser.read() == b'\xaa':
                if self.start:
                    self.now_time = datetime.datetime.now()
                    self.delta_time = (self.now_time - self.pre_time).microseconds / 1e6
                    self.pre_time = self.now_time
                else:
                    self.pre_time = datetime.datetime.now()
                cou = 0
                self.c = self.c + 1
                while True:
                    t = self.ser.read()
                    cou = cou + 1
                    if t != b'\xee':
                        if 0 <= cou - 1 < self.frame_len:
                            self.data[cou - 1] = t
                    else:
                        t = self.ser.read()
                        cou = cou + 1
                        if t == b'\xee':
                            cou = cou - 2
                            break
                        else:
                            if 0 <= cou - 2 < self.frame_len:
                                self.data[cou - 2] = b'\xee'
                            if 0 <= cou - 1 < self.frame_len:
                                self.data[cou - 1] = t
                if cou != self.frame_len:
                    self.wa = self.wa + 1
                    print(str(self.wa) + ' in ' + str(self.c))
                else:
                    if self.start:
                        if self.filled:
                            self.ser.flushInput()
                            return self.data
                        else:
                            self.acc_filter(self.get_acc(self.data))
                self.start = True

    def data_offset(self, cnt):
        # global ser, wa, c, frame_len, data, pre_time, now_time, delta_time, start
        # global acc_offset_x, acc_offset_y, acc_offset_z, gyro_offset_x, gyro_offset_y, gyro_offset_z
        sum_acc = Acc(0.0, 0.0, 0.0)
        sum_gyro = Gyro(0.0, 0.0, 0.0)
        length = cnt

        while True:
            data_frame = self.read_data()
            t_Acc = Acc()
            t_Gyro = Gyro()
            (t_Acc.x, t_Acc.y, t_Acc.z) = self.get_acc(data_frame)
            (t_Gyro.x, t_Gyro.y, t_Gyro.z, _) = self.get_gyro(data_frame)
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
                self.acc_offset_x = sum_acc.x / length
                self.acc_offset_y = sum_acc.y / length
                self.acc_offset_z = sum_acc.z / length
                self.gyro_offset_x = sum_gyro.x / length
                self.gyro_offset_y = sum_gyro.y / length
                self.gyro_offset_z = sum_gyro.z / length
                print('acc_offset:', self.acc_offset_x, self.acc_offset_y, self.acc_offset_z)
                print('gyro_offset:', self.gyro_offset_x, self.gyro_offset_y, self.gyro_offset_z)
                break


if __name__ == '__main__':
    itg3205 = ITG3205('COM6')
    madgwick = cdll.LoadLibrary('madgwick.so')
    madgwick.MadgwickAHRSupdateIMU.restype = c_float
    madgwick.MadgwickAHRSupdateIMU.argtypes = (c_float, c_float, c_float, c_float, c_float, c_float, c_float)
    madgwick.get_theta.restype = c_float
    madgwick.get_hori.restype = c_float
    madgwick.get_ver.restype = c_float
    madgwick.get_rx.restype = c_float
    madgwick.get_vx.restype = c_float
    madgwick.get_vy.restype = c_float
    madgwick.get_vz.restype = c_float

    itg3205_2 = ITG3205('COM7')
    madgwick_2 = cdll.LoadLibrary('madgwick2.so')
    madgwick_2.MadgwickAHRSupdateIMU.restype = c_float
    madgwick_2.MadgwickAHRSupdateIMU.argtypes = (c_float, c_float, c_float, c_float, c_float, c_float, c_float)
    madgwick_2.get_theta.restype = c_float
    madgwick_2.get_hori.restype = c_float
    madgwick_2.get_ver.restype = c_float
    madgwick_2.get_rx.restype = c_float
    madgwick_2.get_vx.restype = c_float
    madgwick_2.get_vy.restype = c_float
    madgwick_2.get_vz.restype = c_float

    while True:
        data_frame = itg3205.read_data()
        ax, ay, az = itg3205.get_acc(data_frame)
        gx, gy, gz, _ = itg3205.angular_velocity(data_frame)
        madgwick.MadgwickAHRSupdateIMU(gx, gy, gz, ax, ay, az, itg3205.delta_time)
        # theta = madgwick.get_theta()
        # hori = madgwick.get_hori()
        # ver = madgwick.get_ver()
        # r_x = madgwick.get_rx()
        # print('theta: %.2f hori: %.2f ver: %.2f r_x: %.2f' % (theta, hori, ver, r_x))
        vx = madgwick.get_vx()
        vy = madgwick.get_vy()
        vz = madgwick.get_vz()

        v = np.array([vx, vy, vz])

        data_frame = itg3205_2.read_data()
        ax, ay, az = itg3205_2.get_acc(data_frame)
        gx, gy, gz, _ = itg3205_2.angular_velocity(data_frame)
        madgwick_2.MadgwickAHRSupdateIMU(gx, gy, gz, ax, ay, az, itg3205_2.delta_time)
        # theta_2 = madgwick_2.get_theta()
        # hori_2 = madgwick_2.get_hori()
        # ver_2 = madgwick_2.get_ver()
        # r_x_2 = madgwick_2.get_rx()
        # print('theta_2: %.2f hori_2: %.2f ver_2: %.2f r_x_2: %.2f' % (theta_2, hori_2, ver_2, r_x_2))

        vx_2 = madgwick_2.get_vx()
        vy_2 = madgwick_2.get_vy()
        vz_2 = madgwick_2.get_vz()
        v_2 = np.array([vx_2, vy_2, vz_2])

        print('%.2f' % (180.0 - ITG3205.get_angle(v, v_2)))
        # madgwick.get_q0.restype = c_float
        # madgwick.get_q1.restype = c_float
        # madgwick.get_q2.restype = c_float
        # madgwick.get_q3.restype = c_float
        # q0 = madgwick.get_q0()
        # q1 = madgwick.get_q1()
        # q2 = madgwick.get_q2()
        # q3 = madgwick.get_q3()
        # z = math.atan2(2 * q1 * q2 - 2 * q0 * q3, 2 * q0 * q0 + 2 * q1 * q1 - 1)
        # y = -math.asin(2 * q1 * q3 + 2 * q0 * q2)
        # x = math.atan2(2 * q2 * q3 - 2 * q0 * q1, 2 * q0 * q0 + 2 * q3 * q3 - 1)
        # print('x:%.2f y:%.2f z:%.2f' % (itg3205.rad2angle(x), (itg3205.rad2angle(y)), itg3205.rad2angle(z)))
