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


class ITG3205:
    IMU_KP = 1.5
    IMU_KI = 0.0005

    def __init__(self):
        self.filter_cnt = 0
        self.FILTER_LENGTH = 20
        self.acc_buff = [Acc] * self.FILTER_LENGTH
        self.filled = False
        (self.acc_offset_x, self.acc_offset_y, self.acc_offset_z) = (0,) * 3
        (self.gyro_offset_x, self.gyro_offset_y, self.gyro_offset_z) = (0,) * 3

        self.q0, self.q1, self.q2, self.q3 = (1.0, 0.0, 0.0, 0.0)
        self.his_q0, self.his_q1, self.his_q2, self.his_q3 = (1.0, 0.0, 0.0, 0.0)
        self.angle = Angle()
        self.v = np.array([1.0, 0.0, 0.0])
        # x、y、z轴的比例误差
        self.ex_int, self.ey_int, self.ez_int = (0.0, 0.0, 0.0)

        self.ser = serial.Serial('COM5', 9600)

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

    def bytes2int(self, x):
        return int.from_bytes(x, byteorder='big', signed=True)

    def get_angle(self, x, y):
        lx = np.sqrt(x.dot(x))
        ly = np.sqrt(y.dot(y))
        cos_angle = x.dot(y) / (lx * ly)
        res = np.arccos(cos_angle)
        return res * 180 / np.pi

    def get_gyro(self, data):
        # 温度buff
        temp = self.bytes2int(data[7] + data[6])
        # 三轴角速度buff
        x = self.bytes2int(data[1] + data[0]) - self.gyro_offset_x
        y = self.bytes2int(data[3] + data[2]) - self.gyro_offset_y
        z = self.bytes2int(data[5] + data[4]) - self.gyro_offset_z
        # print(x, y, z)
        return x, y, z, temp

    def angular_velocity(self, data):
        x, y, z, temp = self.get_gyro(data)
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

    def get_acc(self, data):
        ax = self.bytes2int(data[-5] + data[-6]) - self.acc_offset_x
        ay = self.bytes2int(data[-3] + data[-4]) - self.acc_offset_y
        az = self.bytes2int(data[-1] + data[-2]) - self.acc_offset_z
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

    def acc(self, data):
        ax, ay, az = self.acc_filter(data)
        # pitch = math.atan2(ay, math.sqrt(ax * ax + az * az)) * 180 / math.pi
        # roll = math.atan2(-ax, az) * 180 / math.pi
        # print(pitch, roll, end='\r')
        ax = 0.039 * ax
        ay = 0.039 * ay
        az = 0.039 * az
        # print("ax: %.2f ay: %.2f az: %.2f" % (ax, ay, az))
        return ax, ay, az

    def imu_update(self, gx, gy, gz, ax, ay, az):
        # global q0, q1, q2, q3, his_q0, his_q1, his_q2, his_q3, ex_int, ey_int, ez_int, IMU_KP, IMU_KI
        half_T = self.delta_time / 2

        if ax * ay * az == 0:
            print('ax * ay * az is 0')
            return

        norm = math.sqrt(ax * ax + ay * ay + az * az)
        ax /= norm
        ay /= norm
        az /= norm

        vx = (self.his_q1 * self.his_q3 - self.his_q0 * self.his_q2) * 2
        vy = (self.his_q0 * self.his_q1 + self.his_q2 * self.his_q3) * 2
        vz = self.his_q0 * self.his_q0 - self.his_q1 * self.his_q1 - self.his_q2 * self.his_q2 + self.his_q3 * self.his_q3

        # 向量外积再相减得到差分就是误差，两个单位向量的差积即为误差向量
        ex = (ay * vz - az * vy)
        ey = (az * vx - ax * vz)
        ez = (ax * vy - ay * vx)

        # 对误差进行PI计算
        ex_int = self.ex_int + ex * ITG3205.IMU_KI
        ey_int = self.ey_int + ey * ITG3205.IMU_KI
        ez_int = self.ez_int + ez * ITG3205.IMU_KI

        gx = gx + ITG3205.IMU_KP * ex + ex_int
        gy = gy + ITG3205.IMU_KP * ey + ey_int
        gz = gz + ITG3205.IMU_KP * ez + ez_int

        # 四元素的微分方程
        q0 = self.his_q0 + (-self.his_q1 * gx - self.his_q2 * gy - self.his_q3 * gz) * half_T
        q1 = self.his_q1 + (self.his_q0 * gx + self.his_q2 * gz - self.his_q3 * gy) * half_T
        q2 = self.his_q2 + (self.his_q0 * gy - self.his_q1 * gz + self.his_q3 * gx) * half_T
        q3 = self.his_q3 + (self.his_q0 * gz + self.his_q1 * gy - self.his_q2 * gx) * half_T

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
        self.angle.x = math.atan2(2 * q2q3 + 2 * q0q1, -2 * q1q1 - 2 * q2q2 + 1)
        self.angle.y = math.asin(-2 * q1q3 + 2 * q0q2)
        self.angle.z = math.atan2(2 * q1q2 + 2 * q0q3, -2 * q2q2 - 2 * q3q3 + 1)

        print('A:%.2f H:%.2f V:%.2f R_x:%.2f' %
              (self.get_angle(self.v, tv2), self.get_angle(self.v, h_tv2), self.get_angle(self.v, v_tv2), self.angle.x * 57.3), end='\r')

        # 存储更替相应的四元数
        his_q0 = q0
        his_q1 = q1
        his_q2 = q2
        his_q3 = q3

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
            (t_Acc.x, t_Acc.y, t_Acc.z) = self.get_acc(self.data)
            (t_Gyro.x, t_Gyro.y, t_Gyro.z, _) = self.get_gyro(self.data)
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
    itg3205 = ITG3205()

    while True:
        data_frame = itg3205.read_data()
        ax, ay, az = itg3205.acc(itg3205.get_acc(data_frame))
        gx, gy, gz, _ = itg3205.angular_velocity(data_frame)
        itg3205.imu_update(gx, gy, gz, ax, ay, az)
